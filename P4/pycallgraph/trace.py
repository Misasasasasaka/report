"""Generate a function call graph (callgraph.png) using pycallgraph2.

Usage:
    python trace.py
"""
from pycallgraph2 import PyCallGraph, Config
from pycallgraph2.globbing_filter import GlobbingFilter
from pycallgraph2.output import GraphvizOutput

import app  # our demo program

def run_with_callgraph():
    # Configure which modules to trace (only our 'app' module).
    config = Config()
    config.trace_filter = GlobbingFilter(
        include=['app.*'],
        exclude=['*.tests.*', 'site-packages.*', 'builtins.*']
    )

    graphviz = GraphvizOutput(output_file='callgraph.png')

    # Run app.main() under the tracer.
    with PyCallGraph(output=graphviz, config=config):
        app.main()

if __name__ == '__main__':
    run_with_callgraph()
