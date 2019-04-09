# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2019 SciPy Latam Contributors
#
# Licensed under the terms of the MIT License
# (see spyder/__init__.py for details)
# -----------------------------------------------------------------------------
"""Deploy static lektor website build."""

# Standard library imports
import os

# Third party imports
from fabric import Connection


def deploy(branch):
    """Deploy static lektor website build."""

    SERVER = os.environ.get('DEPLOY_SERVER', '')
    USER = os.environ.get('DEPLOY_USER', '')
    PASS = os.environ.get('DEPLOY_PASS', '')

    con = Connection(
        '{user}@{server}'.format(user=USER, server=SERVER),
        connect_kwargs={'password': PASS},
    )

    con.run(
        'cd web;'
        'ls;'
        # 'git fetch origin;'
        # 'git checkout {branch};'
        # 'git pull origin {branch};'
        # ''.format(branch=branch)
    )


if __name__ == '__main__':
    deploy(branch='gh-pages')
