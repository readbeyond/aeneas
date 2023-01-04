# Building aeneas in docker

Additionally, russian [espeak libraries](https://espeak.sourceforge.net/data/) add into container.

### Build

You need to have [Docker](https://docs.docker.com/engine/install/).

Clone this project, cd to this folder and run:
```
$ docker build -t aenaes:latest .
```

### Usage

To check (get the usage message):

```
$ docker run --name aenaes --rm aenaes python -m aeneas.tools.execute_task
```

To compute a synchronization map `map.json` for a pair (`audio.mp3`, `text.txt` in [plain](http://www.readbeyond.it/aeneas/docs/textfile.html#aeneas.textfile.TextFileFormat.PLAIN) text format):

```
$ docker run --name aenaes --rm aenaes python -m aeneas.tools.execute_task \
    audio.mp3 \
    text.txt \
    "task_language=eng|os_task_file_format=json|is_text_type=plain" \
	--presets-word
    map.json
```
