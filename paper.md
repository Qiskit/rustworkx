---
title: 'retworkx: A High-Performance Graph Library for Python'
tags:
  - graph theory
  - Python
  - Rust
  - quantum computing
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
date: 10 September 2021
bibliography: paper.bib
header-includes:
  - \usepackage{multicol}
---

&nbsp;

>> Network and graph analysis is a widely applicable field of research, and Python is a popular language. In _[retworkx](https://github.com/Qiskit/retworkx)_, we provide a high-performance, flexible graph and network analysis library for Python. _retworkx_ is inspired by _NetworkX_ [@SciPyProceedings_11] but addresses many performance concerns of the latter. _retworkx_ is particularly suited for performance-sensitive applications that use graph representations.

# Statement of need

_retworkx_ is a general-purpose graph theory library focused on performance. It wraps low-level Rust code [@Matsakis2014] into a flexible Python API, providing fast implementations for popular graph algorithms.

_retworkx_ originated from the performance demands of the Qiskit compiler [@Qiskit2021]. At first, Qiskit used the _NetworkX_ library [@SciPyProceedings_11] to construct directed acyclic graph (DAG) representations of quantum circuits which the compiler operates on to perform analysis and transformations [@Childs2019]. As the development of Qiskit progressed, the input size of the executed quantum circuits grew, and _NetworkX_ started to become a bottleneck. Hence, _retworkx_ development emerged to cover the graph usage in Qiskit. The library is now also used by other projects [@Ullberg2021; @Jha2021].

# Related work

To address the performance issues in Qiskit, we explored several graph library alternatives. _igraph_ [@Csardi2006], _graphtool_ [@Peixoto2014], and _SNAP_ [@Leskovec2016] are stable Python libraries written in C or C++ that can replace _NetworkX_.

However, there was a strong desire to keep the flexibility that _NetworkX_ provided for exploring and interacting with the graphs, which precluded custom data structures. The investigated graph libraries either had issues integrating with Qiskit or APIs that were too rigid, such that the migration of existing code was more complex than desired. Thus, the main contribution of _retworkx_ is keeping the ease of use of _NetworkX_ without sacrificing performance.

# Graph data structures

_retworkx_ provides two core data structures: `PyGraph` and `PyDiGraph`. They correspond to undirected and directed graphs, respectively. Graphs describe a set of nodes and the edges connecting pairs of those nodes. Internally, _retworkx_ leverages the _petgraph_ library [@bluss2021] to store the graphs and the _PyO3_ library [@Hewitt2021] for the Python bindings.

Nodes and edges of the graph may also be associated with weights. Weights can contain arbitrary data, such as node labels or edge lengths. Any Python object can be a weight, which makes the library flexible because no assumptions are made about the weight types. 

_retworkx_ operates on weights with callbacks. Callbacks are functions that take weights and return statically typed data. They resemble the named attributes in _NetworkX_. Callbacks are beneficial because they bridge the arbitrary stored data with the static types _retworkx_ expects.

A defining characteristic of _retworkx_ graphs is that each node maps to a non-negative integer node index, and similarly, each edge maps to an edge index. Those indices uniquely determine nodes and edges during the graph object's lifetime. Moreover, the indices provide a clear separation between the underlying graph structure and the data associated with weights. We illustrate indices and callbacks usage with an example.

# Use Cases

_retworkx_ is suitable for modeling graphs ranging from a few nodes scaling up to millions. The library is particularly suited for applications that have core routines executing graph algorithms, such as Qiskit. In those applications, the performance of _retworkx_ makes the difference because it reduces computation time considerably.

We demonstrate the library's performance and use cases comparing _retworkx_ to other popular graph libraries^[_SNAP_ was dropped from the benchmarks because its Python wrapper did not contain the required functions] on a benchmark:

| Library   | _retworkx_| _NetworkX_ | _python-igraph_ | _graphtool_ |
|-----------|-----------|------------|-----------------|-------------|
| Version   | 0.10.2    | 2.6.3      | 0.9.6           | 2.43        |

The benchmark is [available on Github](https://github.com/mtreinish/retworkx-comparison-benchmarks)^[https://github.com/mtreinish/retworkx-comparison-benchmarks] for reproducibility. We present results conducted on the same machine running Python 3.9.7, with 128GB of DDR4 RAM @ 3200MHz and Intel(R) Core i7-6900K CPU @ 3.20GHz with eight cores and 16 threads.

## Graph Creation

The first use case is to represent real-world networks by creating graphs with their respective nodes and edges. We compare the time to create graphs representing the USA road network from the 9th DIMACS challenge dataset [@Demetrescu2016]. Each graph contains $\lvert V \rvert = 23,947,347$ nodes and $\lvert E \rvert = 58,333,344$ weighted edges.

![Time to create the USA road network graph with 23,947,347 nodes and 58,333,344 edges.\label{fig:creation}](paper_img/creation.png){ width=50% height=50% }

The results show that _retworkx_ is 3.1x faster than the second best library in the benchmark, _NetworkX_. The performance gap is even wider compared with _igraph_ and _graphtool_, which are at least 5x slower than _retworkx_.


## Shortest Path

The second use case is to calculate the distance among nodes in a graph using shortest-path algorithms. We compare two scenarios. In the first scenario, we calculate the distance between two nodes in the USA road network. In the second scenario, we calculate the distance among all nodes in the City of Rome road network, with the dataset also coming from the 9th DIMACS challenge. The City of Rome network has $\lvert V \rvert = 3,353$ nodes and $\lvert E \rvert = 8,870$ weighted edges.

\begin{multicols}{2}
\begin{figure}
\centering
\includegraphics[width=0.5\textwidth,height=0.5\textheight]{paper_img/single_source_shortest_path.png}
\caption{Time to find the shortest path between two nodes in the USA road network.\label{fig:sssp}}
\end{figure}

\begin{figure}
\centering
\includegraphics[width=0.5\textwidth,height=0.5\textheight]{paper_img/all_pairs.png}
\caption{Time to find the shortest path among all nodes in the City of Rome road network.\label{fig:allpairs}}
\end{figure}
\end{multicols}

TODO. Also talk that _igraph_/_graphtool_ use another algorithm [@Johnson1977].

## Graph Isomorphism

Talk about definition [@Cordella2001], applications [@Raymond2002] and VF2++ [@Juttner2018]. TODO: benchmark on ARG Database from MIVIA group [@DeSanto2003].

![TODO.\label{fig:subgraphisomorphism}](paper_img/subgraph_isomorphism.png){ width=90% height=90% }

# Acknowledgements

We thank Kevin Krsulich for his help in getting _retworkx_ ready for use by Qiskit; Lauren Capelluto and Itoko Toshinari for their continued support and help with code review; and all of the retworkx contributors who have helped the library improve over time.

# References
