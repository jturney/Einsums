#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Run each profiler test on a thread of its own.

The profiler keeps a zone tree per thread and caps the distinct zone names one parent holds
(``--einsums:profile:max-distinct-children``), folding the rest into "(other)". A pytest run of the
whole suite runs these tests in a process whose main thread has already recorded the zones of
every test before them: thousands of uniquely named graphs, past the cap. A section this file
opens under that root was folded away and every check that looked for it by name failed. A fresh
thread starts with an empty root and its own event buffer, so a test sees what it recorded and
nothing else, whatever ran before it in the process.
"""

import threading

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    arguments = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
    raised = []

    def run():
        try:
            pyfuncitem.obj(**arguments)
        except BaseException as error:  # noqa: BLE001 - re-raised on the test's own thread below
            raised.append(error)

    thread = threading.Thread(target=run, name=f"pytest:{pyfuncitem.name}")
    thread.start()
    thread.join()
    if raised:
        raise raised[0]
    return True
