# Download

FROM=https://dl.cispa.de/s/fgL5StLMpKoa6CE/download/eval_data.zip
wget ${FROM} -O eval_data.zip
unzip eval_data.zip
rm eval_data.zip
rm -rf __MACOSX
