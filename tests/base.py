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

import fixtures
import testtools


class TestCase(testtools.TestCase):

    def setUp(self):
        super(TestCase, self).setUp()
        stdout = self.useFixture(fixtures.StringStream('stdout')).stream
        self.useFixture(fixtures.MonkeyPatch('sys.stdout', stdout))
        stderr = self.useFixture(fixtures.StringStream('stderr')).stream
        self.useFixture(fixtures.MonkeyPatch('sys.stderr', stderr))
        self.useFixture(fixtures.LoggerFixture(nuke_handlers=False,
                                               level=None))
