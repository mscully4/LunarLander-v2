#! /bin/bash

FILES=""
for file in *.mp4; do
    FILES+="${file} \+ ";
done

FILES=${FILES::-3}
eval "mkvmerge -o outfile.mkv $FILES"