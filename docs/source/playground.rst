.. _rustworkx_playground:

####################
Rustworkx Playground
####################

Welcome to the Rustworkx Playground! This is an interactive environment where you can
experiment with Rustworkx on your browser.

.. replite::
   :kernel: python
   :height: 600px
   :prompt: Try it!
   :prompt_color: #6929c4

   import piplite
   await piplite.install(["rustworkx", "matplotlib"])

   import rustworkx as rx
   import rustworkx.visualization as rxviz
   import matplotlib.pyplot as plt

   pet_graph = rx.generators.generalized_petersen_graph(5, 2)
   layout = rx.shell_layout(pet_graph, nlist=[[0, 1, 2, 3, 4], [6, 7, 8, 9, 5]])
   
   plt.ioff(); plt.figure(figsize=(2, 2)); # just use mpl_draw directly, this is only for demos
   rxviz.mpl_draw(pet_graph, pos=layout, node_size=100)
   plt.show()

.. note::
   The `rustworkx` version in the playground is not always the latest. Verify the deployed
   version with `rustworkx.__version__`.

.. note::
   The `rustworkx` version in the playground experimental. If you find any issues, please
   report them at https://github.com/Qiskit/rustworkx/issues.