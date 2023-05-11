# music-transformer-rlhf

# Preparing the feedback interface 
## Generate audio clips from MIDI samples
```./generate_mp3_from_midi.sh /path/to/input/directory /path/to/output/directory```

./generate_mp3_from_midi.sh "/Users/gestalt/Desktop/CS 224R/Project/figaro/samples/figaro-expert/max_iter=16000,max_bars=32" "/Users/gestalt/Desktop/CS 224R/Project/music-transformer-rlhf/audio-dataset"


The ```generate_mp3_from_midi.sh``` script will generate audio files for every
MIDI file in the directory that doesn't already have an associated mp3. It uses
VLC and the default AudioUnit DLS synthesizer in macOS with the following command:
```vlc -I dummy "$f" --sout="#transcode{acodec=mp3,ab=192,channels=2,samplerate=44100}:standard{access=file,mux=raw,dst=$output}" vlc://quit```

## Generate web pages for each pair of audio clips

```python generate_pages.py "../audio-dataset" "pages"```
This will generate pages for each pair of audio files. Samples should be named ID-A.mp3 and ID-B.mp3. The corresponding web page will be generated as pages/ID.html.

```index.html``` will redirect to a random page from the pages directory on pageload.





