#!/bin/bash
if [ "$1" == "cola" ]; then
  fileid="1l-_GExV6NlDNYOto6QSFFIDGY85YH-0p"
  filename="cola.tar.gz"
elif [ "$1" == "mnli" ]; then
  fileid="1qqoP6gfocRlLGHXqT2KwKyHLpLf1rGzX"
  filename="mnli.tar.gz"
elif [ "$1" == "squad" ]; then
  fileid="1Ilt7e5-iBnHCj8Q1loT1uTg1fVp6P3UV"
  filename="squad.tar.gz"
elif [ "$1" == "ner" ]; then
  fileid="1ChgLL9iIJ1_xMaPalIOA99pFqVz86ZBe"
  filename="ner.tar.gz"
fi

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

