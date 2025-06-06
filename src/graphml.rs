// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

#![allow(clippy::borrow_as_ptr)]

use std::convert::From;
use std::ffi::OsStr;
use std::fs::File;
use std::io::Cursor;
use std::io::{BufRead, BufReader};
use std::iter::FromIterator;
use std::num::{ParseFloatError, ParseIntError};
use std::path::Path;
use std::str::ParseBoolError;

use flate2::bufread::GzDecoder;
use hashbrown::HashMap;
use indexmap::IndexMap;

use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use quick_xml::name::QName;
use quick_xml::Error as XmlError;
use quick_xml::{Reader, Writer};

use petgraph::algo;
use petgraph::stable_graph::{EdgeIndex, NodeIndex};
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use petgraph::{Directed, EdgeType, Undirected};

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt};
use pyo3::IntoPyObjectExt;
use pyo3::PyErr;

use crate::{digraph::PyDiGraph, graph::PyGraph, StablePyGraph};

pub enum Error {
    Xml(String),
    ParseValue(String),
    NotFound(String),
    UnSupported(String),
    InvalidDoc(String),
}

impl From<XmlError> for Error {
    #[inline]
    fn from(e: XmlError) -> Error {
        Error::Xml(format!("Xml document not well-formed: {}", e))
    }
}

impl From<ParseBoolError> for Error {
    #[inline]
    fn from(e: ParseBoolError) -> Error {
        Error::ParseValue(format!("Failed conversion to 'bool': {}", e))
    }
}

impl From<ParseIntError> for Error {
    #[inline]
    fn from(e: ParseIntError) -> Error {
        Error::ParseValue(format!("Failed conversion to 'int' or 'long': {}", e))
    }
}

impl From<ParseFloatError> for Error {
    #[inline]
    fn from(e: ParseFloatError) -> Error {
        Error::ParseValue(format!("Failed conversion to 'float' or 'double': {}", e))
    }
}

impl From<Error> for PyErr {
    #[inline]
    fn from(error: Error) -> PyErr {
        match error {
            Error::Xml(msg)
            | Error::ParseValue(msg)
            | Error::NotFound(msg)
            | Error::UnSupported(msg)
            | Error::InvalidDoc(msg) => PyException::new_err(msg),
        }
    }
}

fn xml_attribute<'a>(element: &'a BytesStart<'a>, key: &[u8]) -> Result<String, Error> {
    element
        .attributes()
        .find_map(|a| {
            if let Ok(a) = a {
                if a.key == QName(key) {
                    let decoded = a
                        .unescape_value()
                        .map_err(Error::from)
                        .map(|cow_str| cow_str.into_owned());
                    return Some(decoded);
                }
            }
            None
        })
        .unwrap_or_else(|| {
            Err(Error::NotFound(format!(
                "Attribute '{}' not found.",
                String::from_utf8_lossy(key)
            )))
        })
}

#[derive(Clone, Copy)]
enum Domain {
    Node,
    Edge,
    Graph,
    All,
}

enum Type {
    Boolean,
    Int,
    Float,
    Double,
    String,
    Long,
}

#[derive(Clone)]
enum Value {
    Boolean(bool),
    Int(isize),
    Float(f32),
    Double(f64),
    String(String),
    Long(isize),
    UnDefined,
}

impl<'py> IntoPyObject<'py> for Value {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Value::Boolean(val) => val.into_pyobject(py)?.into_bound_py_any(py),
            Value::Int(val) => Ok(val.into_pyobject(py)?.into_any()),
            Value::Float(val) => Ok(val.into_pyobject(py)?.into_any()),
            Value::Double(val) => Ok(val.into_pyobject(py)?.into_any()),
            Value::String(val) => Ok(val.into_pyobject(py)?.into_any()),
            Value::Long(val) => Ok(val.into_pyobject(py)?.into_any()),
            Value::UnDefined => Ok(py.None().into_bound(py)),
        }
    }
}

struct Key {
    name: String,
    ty: Type,
    default: Value,
}

impl Key {
    fn parse(&self, val: String) -> Result<Value, Error> {
        Ok(match self.ty {
            Type::Boolean => Value::Boolean(val.parse()?),
            Type::Int => Value::Int(val.parse()?),
            Type::Float => Value::Float(val.parse()?),
            Type::Double => Value::Double(val.parse()?),
            Type::String => Value::String(val),
            Type::Long => Value::Long(val.parse()?),
        })
    }

    fn set_value(&mut self, val: String) -> Result<(), Error> {
        self.default = self.parse(val)?;
        Ok(())
    }
}

struct Node {
    id: String,
    data: HashMap<String, Value>,
}

struct Edge {
    id: Option<String>,
    source: String,
    target: String,
    data: HashMap<String, Value>,
}

enum Direction {
    Directed,
    UnDirected,
}

struct Graph {
    dir: Direction,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    attributes: HashMap<String, Value>,
}

impl Graph {
    fn new<'a, I>(dir: Direction, default_attrs: I) -> Self
    where
        I: Iterator<Item = &'a Key>,
    {
        Self {
            dir,
            nodes: Vec::new(),
            edges: Vec::new(),
            attributes: HashMap::from_iter(
                default_attrs.map(|key| (key.name.clone(), key.default.clone())),
            ),
        }
    }

    fn add_node<'a, I>(&mut self, element: &'a BytesStart<'a>, default_data: I) -> Result<(), Error>
    where
        I: Iterator<Item = &'a Key>,
    {
        self.nodes.push(Node {
            id: xml_attribute(element, b"id")?,
            data: HashMap::from_iter(
                default_data.map(|key| (key.name.clone(), key.default.clone())),
            ),
        });

        Ok(())
    }

    fn add_edge<'a, I>(&mut self, element: &'a BytesStart<'a>, default_data: I) -> Result<(), Error>
    where
        I: Iterator<Item = &'a Key>,
    {
        self.edges.push(Edge {
            id: xml_attribute(element, b"id").ok(),
            source: xml_attribute(element, b"source")?,
            target: xml_attribute(element, b"target")?,
            data: HashMap::from_iter(
                default_data.map(|key| (key.name.clone(), key.default.clone())),
            ),
        });

        Ok(())
    }

    fn last_node_set_data(&mut self, key: &Key, val: String) -> Result<(), Error> {
        if let Some(node) = self.nodes.last_mut() {
            node.data.insert(key.name.clone(), key.parse(val)?);
        }

        Ok(())
    }

    fn last_edge_set_data(&mut self, key: &Key, val: String) -> Result<(), Error> {
        if let Some(edge) = self.edges.last_mut() {
            edge.data.insert(key.name.clone(), key.parse(val)?);
        }

        Ok(())
    }
}

impl<'py> IntoPyObject<'py> for Graph {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        macro_rules! make_graph {
            ($graph:ident) => {
                let mut mapping = HashMap::with_capacity(self.nodes.len());
                for mut node in self.nodes {
                    // Write the node id from GraphML doc into the node data payload
                    // since in rustworkx nodes are indexed by an unsigned integer and
                    // not by a hashable String.
                    node.data
                        .insert(String::from("id"), Value::String(node.id.clone()));
                    mapping.insert(node.id, $graph.add_node(node.data.into_py_any(py)?));
                }

                for mut edge in self.edges {
                    match (mapping.get(&edge.source), mapping.get(&edge.target)) {
                        (Some(&source), Some(&target)) => {
                            // Write the edge id from GraphML doc into the edge data payload
                            // since in rustworkx edges are indexed by an unsigned integer and
                            // not by a hashable String.
                            if let Some(id) = edge.id {
                                edge.data.insert(String::from("id"), Value::String(id));
                            }
                            $graph.add_edge(source, target, edge.data.into_py_any(py)?);
                        }
                        _ => {
                            // We skip an edge if one of its endpoints was not added earlier in the graph.
                        }
                    }
                }
            };
        }

        match self.dir {
            Direction::UnDirected => {
                let mut graph =
                    StablePyGraph::<Undirected>::with_capacity(self.nodes.len(), self.edges.len());
                make_graph!(graph);

                let out = PyGraph {
                    graph,
                    node_removed: false,
                    multigraph: true,
                    attrs: self.attributes.into_py_any(py)?,
                };

                Ok(out.into_pyobject(py)?.into_any())
            }
            Direction::Directed => {
                let mut graph =
                    StablePyGraph::<Directed>::with_capacity(self.nodes.len(), self.edges.len());
                make_graph!(graph);

                let out = PyDiGraph {
                    graph,
                    cycle_state: algo::DfsSpace::default(),
                    check_cycle: false,
                    node_removed: false,
                    multigraph: true,
                    attrs: self.attributes.into_py_any(py)?,
                };

                Ok(out.into_pyobject(py)?.into_any())
            }
        }
    }
}

enum State {
    Start,
    Graph,
    Node,
    Edge,
    DataForNode,
    DataForEdge,
    DataForGraph,
    Key,
    DefaultForKey,
}

macro_rules! matches {
    ($expression:expr, $( $pattern:pat_param )|+) => {
        match $expression {
            $( $pattern )|+  => {},
            _ => {
                return Err(Error::InvalidDoc(String::from(
                    "The input xml document doesn't follow the syntax of GraphML language \
                    (or it has features that are not supported by the current version of the parser)."
                )));
            }
        }
    }
}

struct GraphML {
    graphs: Vec<Graph>,
    key_for_nodes: IndexMap<String, Key>,
    key_for_edges: IndexMap<String, Key>,
    key_for_graph: IndexMap<String, Key>,
    key_for_all: IndexMap<String, Key>,
}

impl Default for GraphML {
    fn default() -> Self {
        Self {
            graphs: Vec::new(),
            key_for_nodes: IndexMap::new(),
            key_for_edges: IndexMap::new(),
            key_for_graph: IndexMap::new(),
            key_for_all: IndexMap::new(),
        }
    }
}

impl GraphML {
    fn create_graph<'a>(&mut self, element: &'a BytesStart<'a>) -> Result<(), Error> {
        let dir = match xml_attribute(element, b"edgedefault")?.as_bytes() {
            b"directed" => Direction::Directed,
            b"undirected" => Direction::UnDirected,
            _ => {
                return Err(Error::InvalidDoc(String::from(
                    "Invalid 'edgedefault' attribute.",
                )));
            }
        };

        self.graphs.push(Graph::new(
            dir,
            self.key_for_graph.values().chain(self.key_for_all.values()),
        ));

        Ok(())
    }

    fn add_node<'a>(&mut self, element: &'a BytesStart<'a>) -> Result<(), Error> {
        if let Some(graph) = self.graphs.last_mut() {
            graph.add_node(
                element,
                self.key_for_nodes.values().chain(self.key_for_all.values()),
            )?;
        }

        Ok(())
    }

    fn add_edge<'a>(&mut self, element: &'a BytesStart<'a>) -> Result<(), Error> {
        if let Some(graph) = self.graphs.last_mut() {
            graph.add_edge(
                element,
                self.key_for_edges.values().chain(self.key_for_all.values()),
            )?;
        }

        Ok(())
    }

    fn add_graphml_key<'a>(&mut self, element: &'a BytesStart<'a>) -> Result<Domain, Error> {
        let id = xml_attribute(element, b"id")?;
        let ty = match xml_attribute(element, b"attr.type")?.as_bytes() {
            b"boolean" => Type::Boolean,
            b"int" => Type::Int,
            b"float" => Type::Float,
            b"double" => Type::Double,
            b"string" => Type::String,
            b"long" => Type::Long,
            _ => {
                return Err(Error::InvalidDoc(format!(
                    "Invalid 'attr.type' attribute in key with id={}.",
                    id,
                )));
            }
        };

        let key = Key {
            name: xml_attribute(element, b"attr.name")?,
            ty,
            default: Value::UnDefined,
        };

        match xml_attribute(element, b"for")?.as_bytes() {
            b"node" => {
                self.key_for_nodes.insert(id, key);
                Ok(Domain::Node)
            }
            b"edge" => {
                self.key_for_edges.insert(id, key);
                Ok(Domain::Edge)
            }
            b"graph" => {
                self.key_for_graph.insert(id, key);
                Ok(Domain::Graph)
            }
            b"all" => {
                self.key_for_all.insert(id, key);
                Ok(Domain::All)
            }
            _ => Err(Error::InvalidDoc(format!(
                "Invalid 'for' attribute in key with id={}.",
                id,
            ))),
        }
    }

    fn last_key_set_value(&mut self, val: String, domain: Domain) -> Result<(), Error> {
        let elem = match domain {
            Domain::Node => self.key_for_nodes.last_mut(),
            Domain::Edge => self.key_for_edges.last_mut(),
            Domain::Graph => self.key_for_graph.last_mut(),
            Domain::All => self.key_for_all.last_mut(),
        };

        if let Some((_, key)) = elem {
            key.set_value(val)?;
        }

        Ok(())
    }

    fn last_node_set_data(&mut self, key: &str, val: String) -> Result<(), Error> {
        let key = match self.key_for_all.get(key) {
            Some(key) => key,
            None => self
                .key_for_nodes
                .get(key)
                .ok_or_else(|| Error::NotFound(format!("Key '{}' for nodes not found.", key)))?,
        };

        if let Some(graph) = self.graphs.last_mut() {
            graph.last_node_set_data(key, val)?;
        }

        Ok(())
    }

    fn last_edge_set_data(&mut self, key: &str, val: String) -> Result<(), Error> {
        let key = match self.key_for_all.get(key) {
            Some(key) => key,
            None => self
                .key_for_edges
                .get(key)
                .ok_or_else(|| Error::NotFound(format!("Key '{}' for edges not found.", key)))?,
        };

        if let Some(graph) = self.graphs.last_mut() {
            graph.last_edge_set_data(key, val)?;
        }

        Ok(())
    }

    fn last_graph_set_attribute(&mut self, key: &str, val: String) -> Result<(), Error> {
        let key = match self.key_for_all.get(key) {
            Some(key) => key,
            None => self
                .key_for_graph
                .get(key)
                .ok_or_else(|| Error::NotFound(format!("Key '{}' for graph not found.", key)))?,
        };

        if let Some(graph) = self.graphs.last_mut() {
            graph.attributes.insert(key.name.clone(), key.parse(val)?);
        }

        Ok(())
    }
    /// Open file compressed with gzip, using the GzDecoder
    /// Returns a quick_xml Reader instance
    fn open_file_gzip<P: AsRef<Path>>(
        path: P,
    ) -> Result<Reader<BufReader<GzDecoder<BufReader<File>>>>, quick_xml::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let gzip_reader = BufReader::new(GzDecoder::new(reader));
        Ok(Reader::from_reader(gzip_reader))
    }

    /// Parse a file written in GraphML format from a BufReader
    ///
    /// The implementation is based on a state machine in order to
    /// accept only valid GraphML syntax (e.g a `<data>` element should
    /// be nested inside a `<node>` element) where the internal state changes
    /// after handling each quick_xml event.
    fn read_graph_from_reader<R: BufRead>(mut reader: Reader<R>) -> Result<GraphML, Error> {
        let mut graphml = GraphML::default();

        let mut buf = Vec::new();
        let mut state = State::Start;
        let mut domain_of_last_key = Domain::Node;
        let mut last_data_key = String::new();

        loop {
            match reader.read_event_into(&mut buf)? {
                Event::Start(ref e) => match e.name() {
                    QName(b"key") => {
                        matches!(state, State::Start);
                        domain_of_last_key = graphml.add_graphml_key(e)?;
                        state = State::Key;
                    }
                    QName(b"default") => {
                        matches!(state, State::Key);
                        state = State::DefaultForKey;
                    }
                    QName(b"graph") => {
                        matches!(state, State::Start);
                        graphml.create_graph(e)?;
                        state = State::Graph;
                    }
                    QName(b"node") => {
                        matches!(state, State::Graph);
                        graphml.add_node(e)?;
                        state = State::Node;
                    }
                    QName(b"edge") => {
                        matches!(state, State::Graph);
                        graphml.add_edge(e)?;
                        state = State::Edge;
                    }
                    QName(b"data") => {
                        matches!(state, State::Node | State::Edge | State::Graph);
                        last_data_key = xml_attribute(e, b"key")?;
                        match state {
                            State::Node => state = State::DataForNode,
                            State::Edge => state = State::DataForEdge,
                            State::Graph => state = State::DataForGraph,
                            _ => {
                                // in all other cases we have already bailed out in `matches`.
                                unreachable!()
                            }
                        }
                    }
                    QName(b"hyperedge") => {
                        return Err(Error::UnSupported(String::from(
                            "Hyperedges are not supported.",
                        )));
                    }
                    QName(b"port") => {
                        return Err(Error::UnSupported(String::from("Ports are not supported.")));
                    }
                    _ => {}
                },
                Event::Empty(ref e) => match e.name() {
                    QName(b"key") => {
                        matches!(state, State::Start);
                        graphml.add_graphml_key(e)?;
                    }
                    QName(b"node") => {
                        matches!(state, State::Graph);
                        graphml.add_node(e)?;
                    }
                    QName(b"edge") => {
                        matches!(state, State::Graph);
                        graphml.add_edge(e)?;
                    }
                    QName(b"port") => {
                        return Err(Error::UnSupported(String::from("Ports are not supported.")));
                    }
                    _ => {}
                },
                Event::End(ref e) => match e.name() {
                    QName(b"key") => {
                        matches!(state, State::Key);
                        state = State::Start;
                    }
                    QName(b"default") => {
                        matches!(state, State::DefaultForKey);
                        state = State::Key;
                    }
                    QName(b"graph") => {
                        matches!(state, State::Graph);
                        state = State::Start;
                    }
                    QName(b"node") => {
                        matches!(state, State::Node);
                        state = State::Graph;
                    }
                    QName(b"edge") => {
                        matches!(state, State::Edge);
                        state = State::Graph;
                    }
                    QName(b"data") => {
                        matches!(
                            state,
                            State::DataForNode | State::DataForEdge | State::DataForGraph
                        );
                        match state {
                            State::DataForNode => state = State::Node,
                            State::DataForEdge => state = State::Edge,
                            State::DataForGraph => state = State::Graph,
                            _ => {
                                // in all other cases we have already bailed out in `matches`.
                                unreachable!()
                            }
                        }
                    }
                    _ => {}
                },
                Event::Text(ref e) => match state {
                    State::DefaultForKey => {
                        graphml
                            .last_key_set_value((e.unescape()?).to_string(), domain_of_last_key)?;
                    }
                    State::DataForNode => {
                        graphml.last_node_set_data(&last_data_key, (e.unescape()?).to_string())?;
                    }
                    State::DataForEdge => {
                        graphml.last_edge_set_data(&last_data_key, (e.unescape()?).to_string())?;
                    }
                    State::DataForGraph => {
                        graphml.last_graph_set_attribute(
                            &last_data_key,
                            (e.unescape()?).to_string(),
                        )?;
                    }
                    _ => {}
                },
                Event::Eof => {
                    break;
                }
                _ => {}
            }

            buf.clear();
        }

        Ok(graphml)
    }

    /// Read a graph from a file in the GraphML format
    /// If the the file extension is "graphmlz" or "gz", decompress it on the fly
    fn from_file<P: AsRef<Path>>(path: P, compression: &str) -> Result<GraphML, Error> {
        let extension = path.as_ref().extension().unwrap_or(OsStr::new(""));

        let graph: Result<GraphML, Error> =
            if extension.eq("graphmlz") || extension.eq("gz") || compression.eq("gzip") {
                let reader = Self::open_file_gzip(path)?;
                Self::read_graph_from_reader(reader)
            } else {
                let reader = Reader::from_file(path)?;
                Self::read_graph_from_reader(reader)
            };

        graph
    }
}

/// Read a list of graphs from a file in GraphML format.
///
/// GraphML is a comprehensive and easy-to-use file format for graphs. It consists
/// of a language core to describe the structural properties of a graph and a flexible
/// extension mechanism to add application-specific data.
///
/// For more information see:
/// http://graphml.graphdrawing.org/
///
/// .. note::
///
///     This implementation does not support mixed graphs (directed and undirected edges together),
///     hyperedges, nested graphs, or ports.
///
/// .. note::
///
///     GraphML attributes with `graph` domain are stored in :attr:`~.PyGraph.attrs` field.
///
/// :param str path: The path of the input file to read.
///
/// :return: A list of graphs parsed from GraphML file.
/// :rtype: list[Union[PyGraph, PyDiGraph]]
/// :raises RuntimeError: when an error is encountered while parsing the GraphML file.
#[pyfunction]
#[pyo3(signature=(path, compression=None),text_signature = "(path, /, compression=None)")]
pub fn read_graphml<'py>(
    py: Python<'py>,
    path: &str,
    compression: Option<String>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let graphml = GraphML::from_file(path, &compression.unwrap_or_default())?;

    let mut out = Vec::new();
    for graph in graphml.graphs {
        out.push(graph.into_pyobject(py)?)
    }

    Ok(out)
}

pub fn to_graphml<'py, Ty: EdgeType>(
    py: Python<'py>,
    graph: &StablePyGraph<Ty>,
    graph_attrs: Option<PyObject>,
    node_attrs: Option<PyObject>,
    edge_attrs: Option<PyObject>,
) -> PyResult<String> {
    let mut writer = Writer::new(Cursor::new(Vec::new()));

    // XML Declaration
    writer.write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))?;

    // <graphml> root element
    let mut graphml_start = BytesStart::new("graphml");
    graphml_start.push_attribute(("xmlns", "http://graphml.graphdrawing.org/xmlns"));
    graphml_start.push_attribute(("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance"));
    graphml_start.push_attribute((
        "xsi:schemaLocation",
        "http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd",
    ));
    writer.write_event(Event::Start(graphml_start))?;

    // Cache node attributes
    let node_attr_cache: Option<Vec<(NodeIndex, Bound<'py, PyDict>)>> =
        if let Some(ref callback) = node_attrs {
            Some(
                graph
                    .node_indices()
                    .map(|node| {
                        let weight = &graph[node];
                        let attrs: PyObject = callback.call1(py, (weight,))?;
                        let dict: Bound<'py, PyDict> = attrs.downcast_bound(py)?.clone();
                        Ok((node, dict))
                    })
                    .collect::<PyResult<Vec<_>>>()?,
            )
        } else {
            None
        };

    // Collect node attribute keys and types from cached data
    let mut node_keys = HashMap::new();
    if let Some(ref cache) = node_attr_cache {
        for (_, dict) in cache {
            for item in PyDictMethods::items(dict).iter() {
                let key = item.get_item(0).unwrap();
                let value = item.get_item(1).unwrap();
                let key_str = key.to_string();
                let attr_type = if value.is_instance_of::<PyBool>() {
                    "boolean"
                } else if value.is_instance_of::<PyInt>() {
                    "long"
                } else if value.is_instance_of::<PyFloat>() {
                    "double"
                } else {
                    "string"
                };
                node_keys
                    .entry(key_str)
                    .and_modify(|t: &mut String| {
                        if *t != attr_type && *t != "string" {
                            *t = "string".to_string();
                        }
                    })
                    .or_insert(attr_type.to_string());
            }
        }
    }

    // Cache edge attributes
    let edge_attr_cache: Option<Vec<(EdgeIndex, Bound<'py, PyDict>)>> =
        if let Some(ref callback) = edge_attrs {
            Some(
                graph
                    .edge_references()
                    .map(|edge| {
                        let weight = edge.weight();
                        let attrs: PyObject = callback.call1(py, (weight,))?;
                        let dict: Bound<'py, PyDict> = attrs.downcast_bound(py)?.clone();
                        Ok((edge.id(), dict))
                    })
                    .collect::<PyResult<Vec<_>>>()?,
            )
        } else {
            None
        };

    // Collect edge attribute keys and types from cached data
    let mut edge_keys = HashMap::new();
    if let Some(ref cache) = edge_attr_cache {
        for (_, dict) in cache {
            for item in PyDictMethods::items(dict).iter() {
                let key = item.get_item(0).unwrap();
                let value = item.get_item(1).unwrap();
                let key_str = key.to_string();
                let attr_type = if value.is_instance_of::<PyBool>() {
                    "boolean"
                } else if value.is_instance_of::<PyInt>() {
                    "long"
                } else if value.is_instance_of::<PyFloat>() {
                    "double"
                } else {
                    "string"
                };
                edge_keys
                    .entry(key_str)
                    .and_modify(|t: &mut String| {
                        if *t != attr_type && *t != "string" {
                            *t = "string".to_string();
                        }
                    })
                    .or_insert(attr_type.to_string());
            }
        }
    }

    // Collect graph attribute keys and types
    let mut graph_keys = HashMap::new();
    let graph_attr_dict: Option<Bound<'py, PyDict>> = if let Some(ref callback) = graph_attrs {
        let attrs: PyObject = callback.call0(py)?;
        let dict = attrs.downcast_bound(py)?.clone();
        for item in PyDictMethods::items(&dict).iter() {
            let key = item.get_item(0).unwrap();
            let value = item.get_item(1).unwrap();
            let key_str = key.to_string();
            let attr_type = if value.is_instance_of::<PyBool>() {
                "boolean"
            } else if value.is_instance_of::<PyInt>() {
                "long"
            } else if value.is_instance_of::<PyFloat>() {
                "double"
            } else {
                "string"
            };
            graph_keys.insert(key_str, attr_type.to_string());
        }
        Some(dict)
    } else {
        None
    };

    // Write <key> elements for graph attributes
    for (key, attr_type) in &graph_keys {
        let mut key_start = BytesStart::new("key");
        key_start.push_attribute(("id", key.as_str()));
        key_start.push_attribute(("for", "graph"));
        key_start.push_attribute(("attr.name", key.as_str()));
        key_start.push_attribute(("attr.type", attr_type.as_str()));
        writer.write_event(Event::Empty(key_start))?;
    }

    // Write <key> for node id
    let mut key_start = BytesStart::new("key");
    key_start.push_attribute(("id", "id"));
    key_start.push_attribute(("for", "node"));
    key_start.push_attribute(("attr.name", "id"));
    key_start.push_attribute(("attr.type", "string"));
    writer.write_event(Event::Empty(key_start))?;

    // Write <key> elements for node attributes
    for (key, attr_type) in &node_keys {
        let mut key_start = BytesStart::new("key");
        key_start.push_attribute(("id", key.as_str()));
        key_start.push_attribute(("for", "node"));
        key_start.push_attribute(("attr.name", key.as_str()));
        key_start.push_attribute(("attr.type", attr_type.as_str()));
        writer.write_event(Event::Empty(key_start))?;
    }

    // Write <key> for edge id
    let mut key_start = BytesStart::new("key");
    key_start.push_attribute(("id", "id"));
    key_start.push_attribute(("for", "edge"));
    key_start.push_attribute(("attr.name", "id"));
    key_start.push_attribute(("attr.type", "string"));
    writer.write_event(Event::Empty(key_start))?;

    // Write <key> elements for edge attributes
    for (key, attr_type) in &edge_keys {
        let mut key_start = BytesStart::new("key");
        key_start.push_attribute(("id", key.as_str()));
        key_start.push_attribute(("for", "edge"));
        key_start.push_attribute(("attr.name", key.as_str()));
        key_start.push_attribute(("attr.type", attr_type.as_str()));
        writer.write_event(Event::Empty(key_start))?;
    }

    // Write <graph> element
    let edgedefault = if Ty::is_directed() {
        "directed"
    } else {
        "undirected"
    };
    let mut graph_start = BytesStart::new("graph");
    graph_start.push_attribute(("id", "G"));
    graph_start.push_attribute(("edgedefault", edgedefault));
    writer.write_event(Event::Start(graph_start))?;

    // Write graph attributes
    if let Some(dict) = &graph_attr_dict {
        for item in PyDictMethods::items(dict).iter() {
            let key = item.get_item(0).unwrap();
            let value = item.get_item(1).unwrap();
            let key_str = key.to_string();
            let value_str = value.to_string();
            let mut data_start = BytesStart::new("data");
            data_start.push_attribute(("key", key_str.as_str()));
            writer.write_event(Event::Start(data_start))?;
            writer.write_event(Event::Text(BytesText::new(&value_str)))?;
            writer.write_event(Event::End(BytesEnd::new("data")))?;
        }
    }

    // Write nodes using cached attributes
    for node in graph.node_indices() {
        let node_id = format!("n{}", node.index());
        let mut node_start = BytesStart::new("node");
        node_start.push_attribute(("id", node_id.as_str()));
        writer.write_event(Event::Start(node_start))?;

        // Write node id as data
        let mut data_start = BytesStart::new("data");
        data_start.push_attribute(("key", "id"));
        writer.write_event(Event::Start(data_start))?;
        writer.write_event(Event::Text(BytesText::new(&node_id)))?;
        writer.write_event(Event::End(BytesEnd::new("data")))?;

        // Write node attributes
        if let Some(ref cache) = node_attr_cache {
            if let Some((_, dict)) = cache.iter().find(|(n, _)| *n == node) {
                for item in PyDictMethods::items(dict).iter() {
                    let key = item.get_item(0).unwrap();
                    let value = item.get_item(1).unwrap();
                    let key_str = key.to_string();
                    let value_str = value.to_string();
                    let mut data_start = BytesStart::new("data");
                    data_start.push_attribute(("key", key_str.as_str()));
                    writer.write_event(Event::Start(data_start))?;
                    writer.write_event(Event::Text(BytesText::new(&value_str)))?;
                    writer.write_event(Event::End(BytesEnd::new("data")))?;
                }
            }
        }
        writer.write_event(Event::End(BytesEnd::new("node")))?;
    }

    // Write edges using cached attributes
    if let Some(ref cache) = edge_attr_cache {
        for (edge_id, dict) in cache {
            if let Some((source, target)) = graph.edge_endpoints(*edge_id) {
                let source_id = format!("n{}", source.index());
                let target_id = format!("n{}", target.index());
                let edge_id_str = format!("e{}", edge_id.index());
                let mut edge_start = BytesStart::new("edge");
                edge_start.push_attribute(("id", edge_id_str.as_str()));
                edge_start.push_attribute(("source", source_id.as_str()));
                edge_start.push_attribute(("target", target_id.as_str()));
                writer.write_event(Event::Start(edge_start))?;

                // Write edge id as data
                let mut data_start = BytesStart::new("data");
                data_start.push_attribute(("key", "id"));
                writer.write_event(Event::Start(data_start))?;
                writer.write_event(Event::Text(BytesText::new(&edge_id_str)))?;
                writer.write_event(Event::End(BytesEnd::new("data")))?;

                // Write edge attributes
                for item in PyDictMethods::items(dict).iter() {
                    let key = item.get_item(0).unwrap();
                    let value = item.get_item(1).unwrap();
                    let key_str = key.to_string();
                    let value_str = value.to_string();
                    let mut data_start = BytesStart::new("data");
                    data_start.push_attribute(("key", key_str.as_str()));
                    writer.write_event(Event::Start(data_start))?;
                    writer.write_event(Event::Text(BytesText::new(&value_str)))?;
                    writer.write_event(Event::End(BytesEnd::new("data")))?;
                }
                writer.write_event(Event::End(BytesEnd::new("edge")))?;
            }
        }
    }

    // Close <graph> and <graphml>
    writer.write_event(Event::End(BytesEnd::new("graph")))?;
    writer.write_event(Event::End(BytesEnd::new("graphml")))?;

    let result = writer.into_inner().into_inner();
    Ok(String::from_utf8(result).unwrap())
}

/// Serialize a graph to GraphML format as a string.
///
/// This function converts a `PyGraph` or `PyDiGraph` object into a GraphML string representation.
/// Optional callbacks can be provided to specify attributes for the graph, nodes, and edges.
///
/// Args:
///     graph: The input graph (`PyGraph` or `PyDiGraph`) to serialize.
///     graph_attrs (callable, optional): A callback function that returns a dictionary of graph attributes.
///         The function should take no arguments and return a dict.
///     node_attrs (callable, optional): A callback function that returns a dictionary of node attributes.
///         The function should take a node object as an argument and return a dict.
///     edge_attrs (callable, optional): A callback function that returns a dictionary of edge attributes.
///         The function should take an edge object as an argument and return a dict.
///
/// Returns:
///     str: A string containing the GraphML representation of the graph.
///
/// Raises:
///     TypeError: If the provided graph is neither a `PyGraph` nor a `PyDiGraph`.
///     RuntimeError: If an error occurs during serialization, such as invalid callback outputs.
///
/// Example:
///     >>> import rustworkx as rwx
///     >>> g = rwx.PyGraph()
///     >>> g.add_node("A")
///     >>> g.add_node("B")
///     >>> g.add_edge(0, 1, "edge_data")
///     >>> def node_attrs(node):
///     ...     return {"label": str(node)}
///     >>> def edge_attrs(edge):
///     ...     return {"weight": edge}
///     >>> graphml_str = rwx.write_graphml(g, node_attrs=node_attrs, edge_attrs=edge_attrs)
///     >>> print(graphml_str)
#[pyfunction]
#[pyo3(signature = (graph, graph_attrs=None, node_attrs=None, edge_attrs=None), text_signature = "(graph, /, graph_attrs=None, node_attrs=None, edge_attrs=None)")]
pub fn write_graphml<'py>(
    py: Python<'py>,
    graph: Bound<'py, PyAny>,
    graph_attrs: Option<PyObject>,
    node_attrs: Option<PyObject>,
    edge_attrs: Option<PyObject>,
) -> PyResult<String> {
    if let Ok(pygraph) = graph.extract::<PyGraph>() {
        let stable_graph: StablePyGraph<petgraph::Undirected> = pygraph.graph.clone();
        crate::to_graphml(py, &stable_graph, graph_attrs, node_attrs, edge_attrs)
    } else if let Ok(pydigraph) = graph.extract::<PyDiGraph>() {
        let stable_graph: StablePyGraph<petgraph::Directed> = pydigraph.graph.clone();
        crate::to_graphml(py, &stable_graph, graph_attrs, node_attrs, edge_attrs)
    } else {
        Err(PyException::new_err(
            "Unsupported graph type: must be PyGraph or PyDiGraph",
        ))
    }
}
