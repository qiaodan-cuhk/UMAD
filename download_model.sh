
curl -LsSf https://hf.co/cli/install.sh | bash

MODEL_DIR="${MODEL_DIR:-$HOME/models}"
mkdir -p "$MODEL_DIR"

echo "Use Hugging Face CLI or ModelScope to fetch the model you need."
echo "Examples:"
echo "  huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir \"$MODEL_DIR/qwen2.5-0.5b-instruct\" --local-dir-use-symlinks False"
echo "  huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir \"$MODEL_DIR/qwen3-4b-instruct-2507\" --local-dir-use-symlinks False"
echo "  huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-8B --local-dir \"$MODEL_DIR/ds-r1-distill-llama-8b\" --local-dir-use-symlinks False"
echo "  modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir \"$MODEL_DIR/qwen2.5-7b-instruct\""
echo
echo "For gated models, authenticate with your Hugging Face account before downloading."
