.. _rustworkx_playground:

####################
Rustworkx Playground
####################

Welcome to the Rustworkx Playground! This is an interactive environment where you can
experiment with Rustworkx on your browser.

.. replite::
   :kernel: python
   :height: 600px
   :prompt: Try Rustworkx!
   :prompt_color: #6929c4

   import piplite
   await piplite.install("rustworkx")

   import rustworkx as rx
   import rustworkx.visualization as rxviz
   import matplotlib.pyplot as plt

   pet_graph = rx.generators.generalized_petersen_graph(5, 2)
   pet_layout = rx.shell_layout(pet_graph, nlist=[[0, 1, 2, 3, 4], [6, 7, 8, 9, 5]])
   rxviz.mpl_draw(pet_graph, pos=pet_layout)
   plt.draw()

.. note::
   The `rustworkx` version in the playground is not always the latest. Verify the deployed
   version with `rustworkx.__version__`.

.. note::
   The `rustworkx` version in the playground experimental. If you find any issues, please
   report them at https://github.com/Qiskit/rustworkx/issues.