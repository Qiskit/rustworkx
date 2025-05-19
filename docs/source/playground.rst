.. _rustworkx_visualization:

####################
Rustworkx Playground
####################

Rustworkx Playground
=================

Welcome to the Rustworkx Playground! This is an interactive environment where you can
experiment with Rustworkx on your browser.

.. replite::
   :toctree: apiref
   :kernel: python
   :height: 600px
   :prompt: Try Rustworkx!
   :prompt_color: #dc3545

   import piplite
   await piplite.install("rustworkx")

   import rustworkx as rx
   import rustworkx.visualization as rxviz
   import matplotlib as mpl

   pet_graph = rustworkx.generators.generalized_petersen_graph(5, 2)
   pet_layout = rustworkx.shell_layout(graph, nlist=[[0, 1, 2, 3, 4], [6, 7, 8, 9, 5]])
   mpl_draw(pet_graph, pos=pet_layout)

.. note::
   The `rustworkx` version in the playground is not always the latest. Verify the deployed
   version with `rustworkx.__version__`.

.. note::
   The `rustworkx` version in the playground experimental. If you find any issues, please
   report them at https://github.com/Qiskit/rustworkx/issues.