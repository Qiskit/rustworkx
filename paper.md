---
title: 'retworkx: A High-Performance Graph Library for Python'
tags:
  - graph theory
  - Python
  - Rust
authors:
  - name: Matthew Treinish
    orcid: 0000-0001-9713-2875
    affiliation: 1
  - name: Ivan Carvalho
    orcid: 0000-0002-8257-2103
    affiliation: 2
  - name: Georgios Tsilimigkounakis
    orcid: 0000-0001-6174-0801
    affiliation: 3
  - name: Nahum Sá
    orcid: 0000-0002-3234-8154
    affiliation: 4
affiliations:
 - name: "IBM Quantum, IBM T.J. Watson Research Center, Yorktown Heights, USA \\newline"
   index: 1
 - name: "University of British Columbia, Kelowna, Canada \\newline"
   index: 2
 - name: "National Technical University of Athens, Athens, Greece \\newline"
   index: 3
 - name: "Centro Brasileiro de Pesquisas Físicas, Rio de Janeiro, Brazil"
   index: 4
date: 3 January 2022
bibliography: paper.bib
header-includes:
  - \usepackage{multicol}
---

>> In _[retworkx](https://github.com/Qiskit/retworkx)_, we provide a high-performance, flexible graph library for Python. _retworkx_ is inspired by _NetworkX_ but addresses many performance concerns of the latter. _retworkx_ is written in Rust and is particularly suited for performance-sensitive applications that use graph representations.

# Statement of need

_retworkx_ is a general-purpose graph theory library focused on performance. It wraps low-level Rust code [@Matsakis2014] into a flexible Python API, providing fast implementations for popular graph algorithms.

_retworkx_ originated from the performance demands of the Qiskit compiler [@Qiskit2021]. At first, Qiskit used the _NetworkX_ library [@SciPyProceedings_11] to construct directed acyclic graph (DAG) representations of quantum circuits which the compiler operates on to perform analysis and transformations [@Childs2019]. As the development of Qiskit progressed, the input size of the executed quantum circuits grew, and _NetworkX_ started to become a bottleneck. Hence, _retworkx_ development emerged to cover the graph usage in Qiskit. 

# Related work
  
To address the performance issues in Qiskit, we explored several graph library alternatives. _igraph_ [@Csardi2006], _graph-tool_ [@Peixoto2014], and _SNAP_ [@Leskovec2016] are Python libraries written in C or C++ that can replace _NetworkX_. TODO: _Networkit_ [@Staudt2016] and _SageMath_'s graph theory module [@Sagemath2020]. 

However, there was a strong desire to keep the flexibility that _NetworkX_ provided for exploring and interacting with the graphs, which precluded custom application-specific graph data structures. The graph libraries mentioned above either had issues integrating with Qiskit or APIs that were too rigid, such that the migration of existing code was more complex than desired. Thus, the main contribution of _retworkx_ is keeping the ease of use of _NetworkX_ without sacrificing performance.

# Graph data structures

_retworkx_ provides two core data structures: `PyGraph` and `PyDiGraph`. They correspond to undirected and directed graphs, respectively. Graphs describe a set of nodes and the edges connecting pairs of those nodes. Internally, _retworkx_ leverages the _petgraph_ library [@bluss2021] to store the graphs using an adjacency list model and the _PyO3_ library [@Hewitt2021] for the Python bindings.

Nodes and edges of the graph may also be associated with weights. Weights can contain arbitrary data, such as node labels or edge lengths. Any Python object can be a weight, which makes the library flexible because no assumptions are made about the weight types. 

_retworkx_ operates on weights with callbacks. Callbacks are functions that take weights and return statically typed data. They resemble the named attributes in _NetworkX_. Callbacks are beneficial because they bridge the arbitrary stored data with the static types _retworkx_ expects.

A defining characteristic of _retworkx_ graphs is that each node maps to a non-negative integer node index, and similarly, each edge maps to an edge index. Those indices uniquely determine nodes and edges in the graph. Moreover, the indices provide a clear separation between the underlying graph structure and the data associated with weights.

![A Petersen graph, a hexagonal lattice graph, and a binomial tree graph visualized with the **`retworkx.visualization`** module.\label{fig:graphexample}](paper_img/example_graph.png){ width=100% height==100% }

# Use Cases

_retworkx_ is suitable for modeling graphs ranging from a few nodes scaling up to millions. The library is particularly suited for applications that have core routines executing graph algorithms, such as Qiskit. In those applications, the performance of _retworkx_ considerably reduces computation time.

TODO: talk about other projects [@Ullberg2021; @Jha2021; @Bergholm2020].

## Graph Creation and Manipulation

TODO: talk about [@Ji2021].

## Shortest Path

TODO: talk about [@Dijkstra1959ANO] and quantum error correction.

## Subgraph Isomorphism

TODO: talk about [@Cordella2004], [@Juttner2018], and coupling maps.

# Limitations

TODO

# Acknowledgements

We thank Kevin Krsulich for his help in getting _retworkx_ ready for use by Qiskit; Lauren Capelluto and Toshinari Itoko for their continued support and help with code review; and all of the _retworkx_ contributors who have helped the library improve over time.

# References
