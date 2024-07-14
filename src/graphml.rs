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
use std::iter::FromIterator;
use std::num::{ParseFloatError, ParseIntError};
use std::path::Path;
use std::str::ParseBoolError;

use hashbrown::HashMap;
use indexmap::IndexMap;

use quick_xml::events::{BytesStart, Event};
use quick_xml::name::QName;
use quick_xml::Error as XmlError;
use quick_xml::Reader;

use petgraph::algo;
use petgraph::{Directed, Undirected};

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::PyErr;

use crate::{declare_rustworkx_module, digraph::PyDiGraph, graph::PyGraph, StablePyGraph};

declare_rustworkx_module!(read_graphml);

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

impl IntoPy<PyObject> for Value {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            Value::Boolean(val) => val.into_py(py),
            Value::Int(val) => val.into_py(py),
            Value::Float(val) => val.into_py(py),
            Value::Double(val) => val.into_py(py),
            Value::String(val) => val.into_py(py),
            Value::Long(val) => val.into_py(py),
            Value::UnDefined => py.None(),
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

impl IntoPy<PyObject> for Graph {
    fn into_py(self, py: Python) -> PyObject {
        macro_rules! make_graph {
            ($graph:ident) => {
                let mut mapping = HashMap::with_capacity(self.nodes.len());
                for mut node in self.nodes {
                    // Write the node id from GraphML doc into the node data payload
                    // since in rustworkx nodes are indexed by an unsigned integer and
                    // not by a hashable String.
                    node.data
                        .insert(String::from("id"), Value::String(node.id.clone()));
                    mapping.insert(node.id, $graph.add_node(node.data.into_py(py)));
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
                            $graph.add_edge(source, target, edge.data.into_py(py));
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
                    attrs: self.attributes.into_py(py),
                };

                out.into_py(py)
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
                    attrs: self.attributes.into_py(py),
                };

                out.into_py(py)
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

    /// Parse a file written in GraphML format.
    ///
    /// The implementation is based on a state machine in order to
    /// accept only valid GraphML syntax (e.g a `<data>` element should
    /// be nested inside a `<node>` element) where the internal state changes
    /// after handling each quick_xml event.
    fn from_file<P: AsRef<Path>>(path: P) -> Result<GraphML, Error> {
        let mut graphml = GraphML::default();

        let mut buf = Vec::new();
        let mut reader = Reader::from_file(path)?;

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
///     This implementation does not support mixed graphs (directed and unidirected edges together),
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
#[pyo3(text_signature = "(path, /)")]
pub fn read_graphml(py: Python, path: &str) -> PyResult<Vec<PyObject>> {
    let graphml = GraphML::from_file(path)?;

    let mut out = Vec::new();
    for graph in graphml.graphs {
        out.push(graph.into_py(py))
    }

    Ok(out)
}
