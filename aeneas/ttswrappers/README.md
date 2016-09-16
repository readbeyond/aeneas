# aeneas TTS Wrappers 

This Python package contains the Python wrappers for several built-in
TTS engines that can be used by ``aeneas``
for the synthesis step of the forced alignment.

Each TTS wrapper is a subclass of `BaseTTSWrapper`.
The TTS engine can be called using one of these methods:

1. direct Python call
2. Python C extension
3. TTS executable via ``subprocess``

Currently, the available TTS engines are:

* `ESPEAKTTSWrapper` for `eSpeak` (C extension, subprocess)
* `FESTIVALTTSWrapper` for `festival` (subprocess)
* `NuanceTTSWrapper` for `Nuance TTS API` (Python calling remote Nuance API)

Moreover, custom TTS wrappers can be specified at runtime.
The wrapper must be implemented in a `CustomTTSWrapper` class,
subclass of `BaseTTSWrapper`.
See the `aeneas/extra/` directory for examples.

