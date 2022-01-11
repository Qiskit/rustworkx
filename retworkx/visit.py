# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


class StopSearch(Exception):
    """Stop graph traversal"""

    pass


class PruneSearch(Exception):
    """Prune part of the search tree while traversing a graph."""

    pass


class BFSVisitor:
    """A visitor object that is invoked at the event-points inside the
    :func:`~retworkx.bfs_search` algorithm. By default, it performs no
    action, and should be used as a base class in order to be useful.
    """

    def discover_vertex(self, v):
        """
        This is invoked when a vertex is encountered for the first time.
        """
        return

    def finish_vertex(self, v):
        """
        This is invoked on vertex `v` after all of its out edges have been
        added to the search tree and all of the adjacent vertices have been
        discovered, but before the out-edges of the adjacent vertices have
        been examined.
        """
        return

    def tree_edge(self, e):
        """
        This is invoked on each edge as it becomes a member of the edges
        that form the search tree.
        """
        return

    def non_tree_edge(self, e):
        """
        This is invoked on back or cross edges for directed graphs and cross edges
        for undirected graphs.
        """
        return

    def gray_target_edge(self, e):
        """
        This is invoked on the subset of non-tree edges whose target vertex is
        colored gray at the time of examination.
        The color gray indicates that the vertex is currently in the queue.
        """
        return

    def black_target_edge(self, e):
        """
        This is invoked on the subset of non-tree edges whose target vertex is
        colored black at the time of examination.
        The color black indicates that the vertex has been removed from the queue.
        """
        return


class DFSVisitor:
    """A visitor object that is invoked at the event-points inside the
    :func:`~retworkx.dfs_search` algorithm. By default, it performs no
    action, and should be used as a base class in order to be useful.
    """

    def discover_vertex(self, v, t):
        """
        This is invoked when a vertex is encountered for the first time.
        Together we report the discover time of vertex `v`.
        """
        return

    def finish_vertex(self, v, t):
        """
        This is invoked on vertex `v` after `finish_vertex` has been called for all
        the vertices in the DFS-tree rooted at vertex `v`. If vertex `v` is a leaf in
        the DFS-tree, then the `finish_vertex` function is called on `v` after all
        the out-edges of `v` have been examined. Together we report the finish time
        of vertex `v`.
        """
        return

    def tree_edge(self, e):
        """
        This is invoked on each edge as it becomes a member of the edges
        that form the search tree.
        """
        return

    def back_edge(self, e):
        """
        This is invoked on the back edges in the graph.
        For an undirected graph there is some ambiguity between tree edges
        and back edges since the edge :math:`(u, v)` and :math:`(v, u)` are the
        same edge, but both the `tree_edge()` and `back_edge()` functions will be
        invoked. One way to resolve this ambiguity is to record the tree edges,
        and then disregard the back-edges that are already marked as tree edges.
        An easy way to record tree edges is to record predecessors at the
        `tree_edge` event point.
        """
        return

    def forward_or_cross_edge(self, e):
        """
        This is invoked on forward or cross edges in the graph.
        In an undirected graph this method is never called.
        """
        return
