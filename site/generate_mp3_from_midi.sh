#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

input_dir="$1"
output_dir="$2"

if [ ! -d "$input_dir" ]; then
    echo "Input directory does not exist: $input_dir"
    exit 1
fi

if [ ! -d "$output_dir" ]; then
    echo "Output directory does not exist: $output_dir"
    exit 1
fi

for f in "$input_dir"/*.mid; do
    filename="$(basename -- "$f")"
    output="$output_dir/${filename%.mid}.mp3"
    if [ ! -f "$output" ]; then
        /Applications/VLC.app/Contents/MacOS/VLC -I dummy "$f" --sout="#transcode{acodec=mp3,ab=192,channels=2,samplerate=44100}:standard{access=file,mux=raw,dst=$output}" vlc://quit
    else
        echo "Output file $output already exists. Skipping conversion."
    fi
done

