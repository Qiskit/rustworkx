****************************************************
Rustworkx Comparison Benchmarks With Other Libraries
****************************************************

rustworkx is competitive against other popular graph libraries for Python. We compared rustworkx to the igraph, graph-tools and NetworkIt libraries `in a benchmark consisting of four tasks available on Github for reproducibility <https://github.com/mtreinish/retworkx-comparison-benchmarks>`__ . We report the results from a machine with an Intel(R) i9-9900K CPU at 3.60GHz with eight cores, 16 theads, and 32GB of RAM avaialble. 

Graph Creation
==============

The first use benchmark consists of creating graphs with their respective nodes and edges. We compare the time to create graphs representing the USA road network from the 9th DIMACS challenge dataset (Demetrescu et al., 2009).

.. image:: /images/creation.svg

Single Source Shortest Path
===========================

The second benchmark is to calculate the distance two among nodes in a weighted graph. We compare the time to calculate the distance between the first and the last node in the USA road network, with the datta also coming from the 9th DIMACS challenge (Demetrescu et al., 2009). See :func:`~rustworkx.dijkstra_shortest_path_lengths` for more information on the benchmarked function.

.. image:: /images/single_source_shortest_path.svg

All-Pairs Shortest Path
=======================

The third benchmark is to calculate the distance among all nodes in a weighted graph. We compare the time to calculate the distance among all nodes in the City of Rome road network, another dataset from the 9th DIMACS challenge (Demetrescu et al., 2009). See :func:`~rustworkx.all_pairs_dijkstra_path_lengths` for more information on the benchmarked function.

.. image:: /images/all_pairs.svg

Subgraph Isomorphism
====================

Lastly, the fourth benchamrk is about graph isomorphism. We compare the time to answer if pairs of graphs from the ARG Database are subgraph-isomorphic (De Santo et al., 2003). See :func:`~rustworkx.is_subgraph_isomorphic` for more information on the benchmarked function.

.. image:: /images/subgraph_isomorphism.svg

Citation
--------
* `Demetrescu, C., Goldberg, A., & Johnson, D. The Shortest Path Problem: Ninth DIMACS Implementation Challenge. <https://doi.org/10.1090/dimacs/074>`__
* `Santo, M. D., Foggia, P., Sansone, C., & Vento, M. (2003). A large database of graphs and its use for benchmarking graph isomorphism algorithms. Pattern Recognition Letters, 24(8), 1067–1079.  <https://doi.org/10.1016/S0167-8655(02)00253-2>`__
