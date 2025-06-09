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

use std::borrow::{Borrow, Cow};
use std::convert::From;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter};
use std::iter::FromIterator;
use std::num::{ParseFloatError, ParseIntError};
use std::path::Path;
use std::str::ParseBoolError;

use flate2::bufread::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use hashbrown::{HashMap, HashSet};
use indexmap::IndexMap;

use quick_xml::events::{BytesDecl, BytesStart, BytesText, Event};
use quick_xml::name::QName;
use quick_xml::Error as XmlError;
use quick_xml::{Reader, Writer};

use petgraph::algo;
use petgraph::{Directed, EdgeType, Undirected};

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;
use pyo3::PyErr;

use crate::{digraph::PyDiGraph, graph::PyGraph, StablePyGraph};

pub enum Error {
    Xml(String),
    ParseValue(String),
    NotFound(String),
    UnSupported(String),
    InvalidDoc(String),
    IO(String),
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

impl From<std::io::Error> for Error {
    #[inline]
    fn from(e: std::io::Error) -> Error {
        Error::IO(format!("Input/output error: {}", e))
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
            | Error::InvalidDoc(msg)
            | Error::IO(msg) => PyException::new_err(msg),
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

#[pyclass(eq)]
#[derive(Clone, Copy, PartialEq)]
pub enum Domain {
    Node,
    Edge,
    Graph,
    All,
}

impl TryFrom<&[u8]> for Domain {
    type Error = ();

    fn try_from(value: &[u8]) -> Result<Self, ()> {
        match value {
            b"node" => Ok(Domain::Node),
            b"edge" => Ok(Domain::Edge),
            b"graph" => Ok(Domain::Graph),
            b"all" => Ok(Domain::All),
            _ => Err(()),
        }
    }
}

#[pyclass(eq)]
#[derive(Clone, Copy, PartialEq)]
pub enum Type {
    Boolean,
    Int,
    Float,
    Double,
    String,
    Long,
}

impl Into<&'static str> for Type {
    fn into(self) -> &'static str {
        match self {
            Type::Boolean => "boolean",
            Type::Int => "int",
            Type::Float => "float",
            Type::Double => "double",
            Type::String => "string",
            Type::Long => "long",
        }
    }
}

#[derive(Clone, PartialEq)]
enum Value {
    Boolean(bool),
    Int(isize),
    Float(f32),
    Double(f64),
    String(String),
    Long(isize),
    UnDefined,
}

impl Value {
    fn serialize(&self) -> Option<Cow<str>> {
        match self {
            Value::Boolean(val) => Some(Cow::from(val.to_string())),
            Value::Int(val) => Some(Cow::from(val.to_string())),
            Value::Float(val) => Some(Cow::from(val.to_string())),
            Value::Double(val) => Some(Cow::from(val.to_string())),
            Value::String(val) => Some(Cow::from(val)),
            Value::Long(val) => Some(Cow::from(val.to_string())),
            Value::UnDefined => None,
        }
    }

    fn to_id(&self) -> PyResult<&str> {
        match self {
            Value::String(value_str) => Ok(value_str),
            _ => Err(PyException::new_err("Expected string value for id")),
        }
    }
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

impl Value {
    fn from_pyobject<'py>(ob: &Bound<'py, PyAny>, ty: Type) -> PyResult<Self> {
        let value = match ty {
            Type::Boolean => Value::Boolean(ob.extract::<bool>()?),
            Type::Int => Value::Int(ob.extract::<isize>()?),
            Type::Float => Value::Float(ob.extract::<f32>()?),
            Type::Double => Value::Double(ob.extract::<f64>()?),
            Type::String => Value::String(ob.extract::<String>()?),
            Type::Long => Value::Long(ob.extract::<isize>()?),
        };
        Ok(value)
    }
}

impl<'py> FromPyObject<'py> for Value {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(value) = ob.extract::<bool>() {
            return Ok(Value::Boolean(value));
        }
        if let Ok(value) = ob.extract::<isize>() {
            return Ok(Value::Int(value));
        }
        if let Ok(value) = ob.extract::<f32>() {
            return Ok(Value::Float(value));
        }
        if let Ok(value) = ob.extract::<f64>() {
            return Ok(Value::Double(value));
        }
        if let Ok(value) = ob.extract::<String>() {
            return Ok(Value::String(value));
        }
        Ok(Value::UnDefined)
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
    id: Option<String>,
    dir: Direction,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    attributes: HashMap<String, Value>,
}

impl Graph {
    fn new<'a, I>(id: Option<String>, dir: Direction, default_attrs: I) -> Self
    where
        I: Iterator<Item = &'a Key>,
    {
        Self {
            id,
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

    fn into_pyobject(mut self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        macro_rules! make_graph {
            ($graph:ident) => {
                // Write the graph id from GraphML doc into the graph data payload.
                if let Some(id) = self.id {
                    self.attributes.insert(String::from("id"), Value::String(id.clone()));
                }
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

struct GraphElementInfo {
    attributes: HashMap<String, Value>,
    id: Option<String>,
}

impl Default for GraphElementInfo {
    fn default() -> Self {
        Self {
            attributes: HashMap::new(),
            id: None,
        }
    }
}

struct GraphElementInfos<Index> {
    vec: Vec<(Index, GraphElementInfo)>,
    id_taken: HashSet<String>,
}

impl<Index: std::cmp::Eq + std::hash::Hash> GraphElementInfos<Index> {
    fn new() -> Self {
        Self {
            vec: vec![],
            id_taken: HashSet::new(),
        }
    }

    fn insert<'py>(
        &mut self,
        py: Python<'py>,
        index: Index,
        weight: Option<&Py<PyAny>>,
    ) -> PyResult<()> {
        let element_info = weight
            .and_then(|data| {
                data.extract::<std::collections::HashMap<String, Value>>(py)
                    .ok()
                    .map(|mut attributes| -> PyResult<GraphElementInfo> {
                        let id = attributes
                            .remove_entry("id")
                            .map(|(id, value)| -> PyResult<Option<String>> {
                                let value_str = value.to_id()?;
                                if self.id_taken.contains(value_str) {
                                    attributes.insert(id, value);
                                    Ok(None)
                                } else {
                                    self.id_taken.insert(value_str.to_string());
                                    Ok(Some(value_str.to_string()))
                                }
                            })
                            .unwrap_or_else(|| Ok(None))?;
                        Ok(GraphElementInfo {
                            attributes: attributes.into_iter().collect(),
                            id,
                        })
                    })
            })
            .unwrap_or_else(|| Ok(GraphElementInfo::default()))?;
        self.vec.push((index, element_info));
        Ok(())
    }
}

impl Graph {
    fn try_from_stable<'py, Ty: EdgeType>(
        py: Python<'py>,
        dir: Direction,
        pygraph: &StablePyGraph<Ty>,
        attrs: &PyObject,
    ) -> PyResult<Self> {
        let mut attrs: Option<std::collections::HashMap<String, Value>> = attrs.extract(py).ok();
        let id = attrs
            .as_mut()
            .and_then(|attributes| {
                attributes
                    .remove("id")
                    .map(|v| v.to_id().map(|id| id.to_string()))
            })
            .transpose()?;
        let mut graph = Graph::new(id, dir, std::iter::empty());
        if let Some(attributes) = attrs {
            graph.attributes.extend(attributes);
        }
        let mut node_infos = GraphElementInfos::new();
        for node_index in pygraph.node_indices() {
            node_infos.insert(py, node_index, pygraph.node_weight(node_index))?;
        }
        let mut edge_infos = GraphElementInfos::new();
        for edge_index in pygraph.edge_indices() {
            edge_infos.insert(py, edge_index, pygraph.edge_weight(edge_index))?;
        }
        let mut node_ids = HashMap::new();
        let mut fresh_index_counter = 0;
        for (node_index, element_info) in node_infos.vec {
            let id = element_info.id.unwrap_or_else(|| loop {
                let id = format!("n{fresh_index_counter}");
                fresh_index_counter += 1;
                if node_infos.id_taken.contains(&id) {
                    continue;
                }
                node_infos.id_taken.insert(id.clone());
                break id;
            });
            graph.nodes.push(Node {
                id: id.clone(),
                data: element_info.attributes,
            });
            node_ids.insert(node_index, id);
        }
        for (edge_index, element_info) in edge_infos.vec {
            if let Some((source, target)) = pygraph.edge_endpoints(edge_index) {
                let source = node_ids
                    .get(&source)
                    .ok_or(PyException::new_err("Missing source"))?;
                let target = node_ids
                    .get(&target)
                    .ok_or(PyException::new_err("Missing target"))?;
                graph.edges.push(Edge {
                    id: element_info.id,
                    source: source.clone(),
                    target: target.clone(),
                    data: element_info.attributes,
                });
            }
        }
        Ok(graph)
    }
}

impl<'py> TryFrom<&Bound<'py, PyGraph>> for Graph {
    type Error = PyErr;

    fn try_from(value: &Bound<'py, PyGraph>) -> PyResult<Self> {
        let pygraph = value.borrow();
        return Graph::try_from_stable(
            value.py(),
            Direction::UnDirected,
            &pygraph.graph,
            &pygraph.attrs,
        );
    }
}

impl<'py> TryFrom<&Bound<'py, PyDiGraph>> for Graph {
    type Error = PyErr;

    fn try_from(value: &Bound<'py, PyDiGraph>) -> PyResult<Self> {
        let pygraph = value.borrow();
        return Graph::try_from_stable(
            value.py(),
            Direction::Directed,
            &pygraph.graph,
            &pygraph.attrs,
        );
    }
}

impl<'py> FromPyObject<'py> for Graph {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        match ob.downcast::<PyGraph>() {
            Ok(graph) => Graph::try_from(graph),
            Err(_) => Graph::try_from(ob.downcast::<PyDiGraph>()?),
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

/// Given maps from ids to keys, return a map from key name to ids and keys.
fn build_key_name_map<'a>(
    key_for_items: &'a IndexMap<String, Key>,
    key_for_all: &'a IndexMap<String, Key>,
) -> HashMap<String, (&'a String, &'a Key)> {
    // `key_for_items` is iterated before `key_for_all` since last
    // items take precedence in the collected map. Similarly,
    // the map `for_all` take precedence over kind-specific maps in
    // `last_node_set_data`, `last_edge_set_data` and
    // `last_graph_set_attribute`.
    key_for_all
        .iter()
        .chain(key_for_items.iter())
        .map(|(id, key)| (key.name.clone(), (id, key)))
        .collect()
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
            xml_attribute(element, b"id").ok(),
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

    fn get_keys(&self, domain: Domain) -> &IndexMap<String, Key> {
        match domain {
            Domain::Node => &self.key_for_nodes,
            Domain::Edge => &self.key_for_edges,
            Domain::Graph => &self.key_for_graph,
            Domain::All => &self.key_for_all,
        }
    }

    fn get_keys_mut(&mut self, domain: Domain) -> &mut IndexMap<String, Key> {
        match domain {
            Domain::Node => &mut self.key_for_nodes,
            Domain::Edge => &mut self.key_for_edges,
            Domain::Graph => &mut self.key_for_graph,
            Domain::All => &mut self.key_for_all,
        }
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
        let domain: Domain = xml_attribute(element, b"for")?
            .as_bytes()
            .try_into()
            .map_err(|()| {
                Error::InvalidDoc(format!("Invalid 'for' attribute in key with id={}.", id,))
            })?;
        self.get_keys_mut(domain).insert(id, key);
        Ok(domain)
    }

    fn last_key_set_value(&mut self, val: String, domain: Domain) -> Result<(), Error> {
        let elem = self.get_keys_mut(domain).last_mut();

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

    fn write_data<W: std::io::Write>(
        writer: &mut Writer<W>,
        keys: &HashMap<String, (&String, &Key)>,
        data: &HashMap<String, Value>,
    ) -> Result<(), Error> {
        for (key_name, value) in data {
            let (id, key) = keys
                .get(key_name)
                .ok_or_else(|| Error::NotFound(format!("Unknown key {key_name}")))?;
            if key.default == *value {
                continue;
            }
            let mut elem = BytesStart::new("data");
            elem.push_attribute(("key", id.as_str()));
            writer.write_event(Event::Start(elem.borrow()))?;
            if let Some(contents) = value.serialize() {
                writer.write_event(Event::Text(BytesText::new(contents.borrow())))?;
            }
            writer.write_event(Event::End(elem.to_end()))?;
        }
        Ok(())
    }

    fn write_elem_data<W: std::io::Write>(
        writer: &mut Writer<W>,
        keys: &HashMap<String, (&String, &Key)>,
        elem: BytesStart,
        data: &HashMap<String, Value>,
    ) -> Result<(), Error> {
        if data.is_empty() {
            writer.write_event(Event::Empty(elem))?;
            return Ok(());
        }
        writer.write_event(Event::Start(elem.borrow()))?;
        Self::write_data(writer, keys, data)?;
        writer.write_event(Event::End(elem.to_end()))?;
        Ok(())
    }

    fn write_keys<W: std::io::Write>(
        writer: &mut Writer<W>,
        key_for: &str,
        map: &IndexMap<String, Key>,
    ) -> Result<(), quick_xml::Error> {
        for (id, key) in map {
            let mut elem = BytesStart::new("key");
            elem.push_attribute(("id", id.as_str()));
            elem.push_attribute(("for", key_for));
            elem.push_attribute(("attr.name", key.name.as_str()));
            let ty: &str = key.ty.into();
            elem.push_attribute(("attr.type", ty));
            writer.write_event(Event::Start(elem.borrow()))?;
            if let Some(contents) = key.default.serialize() {
                let elem = BytesStart::new("default");
                writer.write_event(Event::Start(elem.borrow()))?;
                writer.write_event(Event::Text(BytesText::new(contents.borrow())))?;
                writer.write_event(Event::End(elem.to_end()))?;
            };
            writer.write_event(Event::End(elem.to_end()))?;
        }
        Ok(())
    }

    fn write_graph_to_writer<W: std::io::Write>(
        &self,
        writer: &mut Writer<W>,
    ) -> Result<(), Error> {
        writer.write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))?;
        let mut elem = BytesStart::new("graphml");
        elem.push_attribute(("xmlns", "http://graphml.graphdrawing.org/xmlns"));
        elem.push_attribute(("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance"));
        elem.push_attribute((
            "xsi:schemaLocation",
            "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd",
        ));
        writer.write_event(Event::Start(elem.borrow()))?;
        Self::write_keys(writer, "node", &self.key_for_nodes)?;
        Self::write_keys(writer, "edge", &self.key_for_edges)?;
        Self::write_keys(writer, "graph", &self.key_for_graph)?;
        Self::write_keys(writer, "all", &self.key_for_all)?;
        let graph_keys: HashMap<String, (&String, &Key)> =
            build_key_name_map(&self.key_for_graph, &self.key_for_all);
        let node_keys: HashMap<String, (&String, &Key)> =
            build_key_name_map(&self.key_for_nodes, &self.key_for_all);
        let edge_keys: HashMap<String, (&String, &Key)> =
            build_key_name_map(&self.key_for_edges, &self.key_for_all);
        for graph in self.graphs.iter() {
            let mut elem = BytesStart::new("graph");
            if let Some(id) = &graph.id {
                elem.push_attribute(("id", id.as_str()));
            }
            let edgedefault = match graph.dir {
                Direction::Directed => "directed",
                Direction::UnDirected => "undirected",
            };
            elem.push_attribute(("edgedefault", edgedefault));
            writer.write_event(Event::Start(elem.borrow()))?;
            Self::write_data(writer, &graph_keys, &graph.attributes)?;
            for node in &graph.nodes {
                let mut elem = BytesStart::new("node");
                elem.push_attribute(("id", node.id.as_str()));
                Self::write_elem_data(writer, &node_keys, elem, &node.data)?;
            }
            for edge in &graph.edges {
                let mut elem = BytesStart::new("edge");
                if let Some(id) = &edge.id {
                    elem.push_attribute(("id", id.as_str()));
                }
                elem.push_attribute(("source", edge.source.as_str()));
                elem.push_attribute(("target", edge.target.as_str()));
                Self::write_elem_data(writer, &edge_keys, elem, &edge.data)?;
            }
            writer.write_event(Event::End(elem.to_end()))?;
        }
        writer.write_event(Event::End(elem.to_end()))?;
        Ok(())
    }

    fn to_file(&self, path: impl AsRef<Path>, compression: &str) -> Result<(), Error> {
        let extension = path.as_ref().extension().unwrap_or(OsStr::new(""));
        if extension.eq("graphmlz") || extension.eq("gz") || compression.eq("gzip") {
            let file = File::create(path)?;
            let buf_writer = BufWriter::new(file);
            let gzip_encoder = GzEncoder::new(buf_writer, Compression::default());
            let mut writer = Writer::new(gzip_encoder);
            self.write_graph_to_writer(&mut writer)?;
            writer.into_inner().finish()?;
        } else {
            let file = File::create(path)?;
            let mut writer = Writer::new(file);
            self.write_graph_to_writer(&mut writer)?;
        }
        Ok(())
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

/// Read a list of graphs from a file in GraphML format and return the pair containing the list of key definitions and the graph.
///
/// Each key definition is a tuple: id, domain, name of the key, type, default value.
#[pyfunction]
#[pyo3(signature=(path, compression=None),text_signature = "(path, /, compression=None)")]
pub fn read_graphml_with_keys<'py>(
    py: Python<'py>,
    path: &str,
    compression: Option<String>,
) -> PyResult<(
    Vec<(String, Domain, String, Type, Bound<'py, PyAny>)>,
    Vec<Bound<'py, PyAny>>,
)> {
    let graphml = GraphML::from_file(path, &compression.unwrap_or_default())?;

    let mut keys = Vec::new();
    for domain in [Domain::Node, Domain::Edge, Domain::Graph, Domain::All] {
        for (id, key) in graphml.get_keys(domain) {
            let default = key.default.clone().into_pyobject(py)?.into_any();
            keys.push((id.clone(), domain, key.name.clone(), key.ty, default));
        }
    }

    let mut out = Vec::new();
    for graph in graphml.graphs {
        out.push(graph.into_pyobject(py)?)
    }

    Ok((keys, out))
}

/// Write a list of graphs to a file in GraphML format given the list of key definitions.
#[pyfunction]
#[pyo3(signature=(graphs, keys, path, compression=None),text_signature = "(graphs, keys, path, /, compression=None)")]
pub fn write_graphml<'py>(
    py: Python<'py>,
    graphs: Vec<Py<PyAny>>,
    keys: Vec<(String, Domain, String, Type, Py<PyAny>)>,
    path: &str,
    compression: Option<String>,
) -> PyResult<()> {
    let mut graphml = GraphML::default();
    for (id, domain, name, ty, default) in keys {
        let bound_default = default.bind(py);
        let default = if bound_default.is_none() {
            Value::UnDefined
        } else {
            Value::from_pyobject(bound_default, ty)?
        };
        graphml
            .get_keys_mut(domain)
            .insert(id, Key { name, ty, default });
    }
    for graph in graphs {
        graphml.graphs.push(Graph::extract_bound(graph.bind(py))?)
    }
    graphml.to_file(path, &compression.unwrap_or_default())?;
    Ok(())
}
