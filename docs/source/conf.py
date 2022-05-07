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
# retworkx documentation build configuration file
#

import sys, os
import subprocess

# General configuration:

project = u'retworkx'
copyright = u'2021, retworkx Contributors'


# The short X.Y version.
version = '0.12.0'
# The full version, including alpha/beta/rc tags.
release = '0.12.0'

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
              'sphinx_reredirects',
             ]
html_static_path = ['_static']
templates_path = ['_templates']
html_css_files = ['style.css', 'custom.css']

pygments_style = 'colorful'

add_module_names = False

modindex_common_prefix = ['retworkx.']

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

if not os.getenv('RETWORKX_DEV_DOCS', None):
    rst_prolog = """
.. raw:: html

    <br><br><br>

""".format(release)
else:
    rst_prolog = """
.. raw:: html

    <br><br><br>

.. note::

    This is the documnetation for the current state of the development branch
    of retworkx. The documentation or APIs here can change prior to being
    released.

"""

# HTML Output Options

html_theme = 'qiskit_sphinx_theme'

html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
}

htmlhelp_basename = 'retworkx'


# Latex options

latex_elements = {}

latex_documents = [
  ('index', 'retworkx.tex', u'retworkx Documentation',
   u'retworkx Contributors', 'manual'),
]

# Texinfo options

texinfo_documents = [
  ('index', 'retworkx', u'retworkx Documentation',
   u'retworkx Contributors', 'retworkx', '',
   'Miscellaneous'),
]

redirects = {}
with open("sources.txt", "r") as fd:
    for source_str in fd:
        redirects[f"stubs/{source_str}"] = f"../apiref/{source_str}"

# Version extensions

def _get_versions(app, config):
    context = config.html_context
    start_version = (0, 8, 0)
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
    if not os.getenv('RETWORKX_DEV_DOCS', None):
        current_version_info = current_version.split('.')
        return ".".join(current_version_info[:-1])
    else:
        return "Development"


def setup(app):
    app.connect('config-inited', _get_versions)
