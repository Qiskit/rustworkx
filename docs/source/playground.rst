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

   print(f"rustworkx version: {rx.__version__}")

.. note::
   The `rustworkx` version in the playground is not always the latest. Verify the deployed
   version with `rustworkx.__version__`.

.. note::
   The `rustworkx` version in the playground experimental. If you find any issues, please
   report them at https://github.com/Qiskit/rustworkx/issues.