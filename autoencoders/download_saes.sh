# Other SAE groups can be found at https://huggingface.co/adamkarvonen/chess_saes and https://huggingface.co/adamkarvonen/othello_saes
# The following 2 groups are fairly large once unzipped - around 150 GB total.

# Check if unzip is installed, and exit if it isn't
if ! command -v unzip &> /dev/null
then
    echo "Error: unzip is not installed. Please install it and rerun the setup script."
    exit 1
fi

wget -O othello-trained_model-layer_5-2024-05-23.zip "https://huggingface.co/adamkarvonen/othello_saes/resolve/main/othello-trained_model-layer_5-2024-05-23.zip?download=true"

unzip othello-trained_model-layer_5-2024-05-23.zip
rm othello-trained_model-layer_5-2024-05-23.zip

wget -O chess-trained_model-layer_5-2024-05-23.zip "https://huggingface.co/adamkarvonen/chess_saes/resolve/main/chess-trained_model-layer_5-2024-05-23.zip?download=true"

unzip chess-trained_model-layer_5-2024-05-23.zip
rm chess-trained_model-layer_5-2024-05-23.zip