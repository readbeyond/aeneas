# Building aeneas in docker

Additionally, russian [espeak libraries](https://espeak.sourceforge.net/data/) add into container.

## Build

You need to have [Docker](https://docs.docker.com/engine/install/).

Clone this project, cd to this folder and run:
```
$ docker build -t aenaes:latest .
```

## Usage

To check (get the usage message):

```
$ docker run --name aenaes --rm aenaes python -m aeneas.tools.execute_task
```

To compute a synchronization map `map.json` for a pair (`/path/to/your/files/audio.mp3`, `/path/to/your/files/text.txt` in [plain](http://www.readbeyond.it/aeneas/docs/textfile.html#aeneas.textfile.TextFileFormat.PLAIN) text format):

```
$ docker run --name aenaes --rm --volume /path/to/your/files:/data aenaes \
    python -m aeneas.tools.execute_task \
    /data/audio.mp3 \
    /data/text.txt \
    "task_language=eng|os_task_file_format=json|is_text_type=plain" \
    map.json \
    --rate --presets-word
```

You will get a file `/path/to/your/files/map.json` as a result.