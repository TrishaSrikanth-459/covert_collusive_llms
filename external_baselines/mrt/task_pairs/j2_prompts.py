"""Optional jailbreak-prefix context for MRT.

The public MRT code imports this module unconditionally, but the smoke/full
runs in this project keep `jailbreak_prefix=False`. An empty list preserves the
default behavior while making the public task release runnable when the private
prompt artifact is absent.
"""

HIDDEN_CONTEXT = []
