# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

#
# rustworkx documentation build configuration file
#

import sys, os
import subprocess

# General configuration:

project = 'rustworkx'
copyright = '2021, rustworkx Contributors'
docs_url_prefix = ""

# The short X.Y version.
version = '0.14.0'
# The full version, including alpha/beta/rc tags.
release = '0.14.0'

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.viewcode',
              'sphinx.ext.extlinks',
              'sphinx.ext.todo',
              'sphinx.ext.viewcode',
              'm2r2',
              'jupyter_sphinx',
              'reno.sphinxext',
              'sphinx.ext.intersphinx',
              'sphinxemoji.sphinxemoji',
              'sphinx_reredirects',
              'qiskit_sphinx_theme',
             ]
templates_path = ['_templates']
extra_css_files = ["overrides.css"]

pygments_style = 'colorful'

add_module_names = False

modindex_common_prefix = ['rustworkx.']

todo_include_todos = True

source_suffix = ['.rst', '.md']

master_doc = 'index'

# Autosummary options
autosummary_generate = True
autosummary_generate_overwrite = False
autoclass_content = 'both'
autodoc_typehints = 'none' # disabled until https://github.com/Qiskit/qiskit_sphinx_theme/issues/21 is fixed

# Intersphinx configuration
intersphinx_mapping = {
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None)
}

# Prepend warning for development docs:

if not os.getenv('RUSTWORKX_DEV_DOCS', None):
    rst_prolog = """
.. raw:: html

    <br><br><br>

""".format(release)
else:
    rst_prolog = """
.. raw:: html

    <br><br><br>

.. note::

    This is the documentation for the current state of the development branch
    of rustworkx. The documentation or APIs here can change prior to being
    released.

"""

# HTML Output Options
html_theme = 'qiskit-ecosystem'
html_title = f"{project} {release}"
htmlhelp_basename = 'rustworkx'

html_theme_options = {
    "disable_ecosystem_logo": True,
}

# Latex options
latex_elements = {}
latex_documents = [
  ('index', 'rustworkx.tex', u'rustworkx Documentation',
   u'rustworkx Contributors', 'manual'),
]

# Jupyter Sphinx options
jupyter_execute_default_kernel = "python3"

# Texinfo options

texinfo_documents = [
  ('index', 'rustworkx', u'rustworkx Documentation',
   u'rustworkx Contributors', 'rustworkx', '',
   'Miscellaneous'),
]

redirects = {}
with open("sources.txt", "r") as fd:
    for source_str in fd:
        redirects[f"stubs/{source_str}"] = f"../apiref/{source_str}"

if os.getenv("RUSTWORKX_LEGACY_DOCS", None) is not None:
    redirects["*"] = "https://www.rustworkx.org/$source.html"
    html_baseurl = "https://www.rustworkx.org/"


# Version extensions

def _get_versions(app, config):
    context = config.html_context
    start_version = (0, 12, 0)
    proc = subprocess.run(['git', 'describe', '--abbrev=0'],
                          capture_output=True)
    proc.check_returncode()
    current_version = proc.stdout.decode('utf8')
    current_version_info = current_version.split('.')
    if current_version_info[0] == '0':
        version_list = [
            '0.%s' % x for x in range(start_version[1],
                                      int(current_version_info[1]) + 1)]
    else:
        #TODO: When 1.0.0 add code to handle 0.x version list
        version_list = []
        pass
    context['version_list'] = version_list
    context['version_label'] = _get_version_label(current_version)


def _get_version_label(current_version):
    if not os.getenv('RUSTWORKX_DEV_DOCS', None):
        current_version_info = current_version.split('.')
        return ".".join(current_version_info[:-1])
    else:
        return "Development"

def avoid_duplicate_in_dispatch(app, obj, bound_method):
    if hasattr(obj, 'dispatch') and hasattr(obj, 'register') and obj.dispatch.__module__ == 'functools':
        # TODO: disable this trick once https://github.com/Qiskit/qiskit_sphinx_theme/issues/21 is fixed
        # Basically, to avoid signatures being duplicated, we want to disable the singledispatch function
        # from Sphinx's autodoc. But if we unregister the function, our jupyter notebook executions
        # will fail. Hence we just trick the check in sphinx/util/inspect/is_singledispatch_function
        # that checks for obj.dispatch.__module__ == 'functools'. This should be harmless as
        # that property is only used on __repr__ and not to dispatch the function itself
        obj.dispatch.__module__ = "rustworkx"


def setup(app):
    app.connect('config-inited', _get_versions)
    app.connect('autodoc-before-process-signature', avoid_duplicate_in_dispatch)
