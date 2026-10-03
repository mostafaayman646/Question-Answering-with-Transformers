"""
╔══════════════════════════════════════════════════════════════════════╗
║              🤖 Question Answering with DistilBERT                 ║
║                    Interactive Inference CLI                        ║
╚══════════════════════════════════════════════════════════════════════╝

Load a fine-tuned DistilBERT model and answer questions against a
curated bank of real-world contexts — or supply your own.
"""

import torch
import json
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast, DistilBertConfig

# ─────────────────────────────────────────────────────────────────────
# 🎨 Terminal colours & helpers
# ─────────────────────────────────────────────────────────────────────
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
MAGENTA = "\033[95m"
RED     = "\033[91m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"

DIVIDER = f"{DIM}{'─' * 70}{RESET}"

# ─────────────────────────────────────────────────────────────────────
# 📚 Curated context bank – diverse real-world paragraphs
# ─────────────────────────────────────────────────────────────────────
CONTEXTS = [
    {
        "title": "🌌 The Solar System",
        "text": (
            "The Solar System consists of the Sun and the astronomical objects "
            "gravitationally bound to it. The eight planets, in order from the Sun, "
            "are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. "
            "Jupiter is the largest planet in the Solar System, with a mass more than "
            "twice that of all other planets combined. Saturn is famous for its "
            "prominent ring system. Earth is the only planet known to support life."
        ),
    },
    {
        "title": "🧬 DNA and Genetics",
        "text": (
            "Deoxyribonucleic acid (DNA) is a molecule composed of two polynucleotide "
            "chains that coil around each other to form a double helix. DNA carries "
            "genetic instructions for the development, functioning, growth, and "
            "reproduction of all known organisms and many viruses. DNA was first "
            "isolated by Friedrich Miescher in 1869. The structure of DNA was "
            "discovered by James Watson and Francis Crick in 1953, using X-ray "
            "diffraction data collected by Rosalind Franklin."
        ),
    },
    {
        "title": "🏛️ Ancient Egypt",
        "text": (
            "Ancient Egypt was a civilization in Northeast Africa concentrated along "
            "the lower reaches of the Nile River. The Great Pyramid of Giza was built "
            "around 2560 BC as a tomb for Pharaoh Khufu. It is the oldest of the "
            "Seven Wonders of the Ancient World and the only one to remain largely "
            "intact. The ancient Egyptians developed a writing system known as "
            "hieroglyphics and made advances in medicine, mathematics, and astronomy."
        ),
    },
    {
        "title": "💻 History of the Internet",
        "text": (
            "The Internet originated as ARPANET in the late 1960s, funded by the "
            "United States Department of Defense. Tim Berners-Lee invented the World "
            "Wide Web in 1989 while working at CERN. The first website was created in "
            "1991 and was dedicated to information about the World Wide Web project "
            "itself. By 2023, over 5.3 billion people worldwide were using the "
            "Internet, representing about 66 percent of the global population."
        ),
    },
    {
        "title": "⚽ FIFA World Cup",
        "text": (
            "The FIFA World Cup is an international football competition contested by "
            "the senior men's national teams of FIFA member associations. The first "
            "World Cup was held in Uruguay in 1930 and was won by the host nation. "
            "Brazil has won the tournament a record five times. The 2022 World Cup was "
            "held in Qatar, where Argentina defeated France in the final, with Lionel "
            "Messi winning the Golden Ball award for the best player of the tournament."
        ),
    },
    {
        "title": "🎵 The Evolution of Music",
        "text": (
            "Music has been a fundamental part of human culture for thousands of years. "
            "The oldest known musical instrument is a flute made from a vulture bone, "
            "found in a cave in Germany and dating back approximately 40,000 years. "
            "Classical music reached its peak during the 18th and 19th centuries with "
            "composers like Mozart, Beethoven, and Bach. The electric guitar, invented "
            "in the 1930s, revolutionized popular music and gave birth to rock and roll."
        ),
    },
    {
        "title": "🧠 Artificial Intelligence",
        "text": (
            "Artificial intelligence (AI) is intelligence demonstrated by machines, "
            "as opposed to natural intelligence displayed by animals and humans. "
            "The term was coined by John McCarthy in 1956 at the Dartmouth Conference. "
            "Modern AI techniques include machine learning, deep learning, and natural "
            "language processing. In 2016, Google DeepMind's AlphaGo defeated the world "
            "champion Go player Lee Sedol, marking a milestone in AI research. "
            "Large language models such as GPT and BERT have transformed how machines "
            "understand and generate human language."
        ),
    },
    {
        "title": "🏔️ Mount Everest",
        "text": (
            "Mount Everest is the highest mountain above sea level on Earth, with a "
            "peak elevation of 8,849 metres (29,032 feet). It is located in the "
            "Mahalangur Himal sub-range of the Himalayas, on the border between Nepal "
            "and Tibet. The first confirmed ascent was made on 29 May 1953 by Edmund "
            "Hillary from New Zealand and Tenzing Norgay, a Sherpa from Nepal. As of "
            "2023, over 6,000 people have successfully reached the summit."
        ),
    },
    {
        "title": "🩺 The Human Heart",
        "text": (
            "The human heart is a muscular organ roughly the size of a closed fist. "
            "It beats approximately 100,000 times per day, pumping about 7,570 litres "
            "of blood throughout the body. The heart has four chambers: the left atrium, "
            "the right atrium, the left ventricle, and the right ventricle. The left "
            "ventricle is the strongest chamber because it must pump blood to the "
            "entire body through the aorta. Heart disease is the leading cause of "
            "death worldwide."
        ),
    },
    {
        "title": "☕ The History of Coffee",
        "text": (
            "Coffee is believed to have been discovered in Ethiopia around the 9th "
            "century by a goat herder named Kaldi, who noticed his goats became "
            "energetic after eating berries from a certain tree. By the 15th century, "
            "coffee was being cultivated in Yemen. The first coffeehouse opened in "
            "Constantinople in 1475. Today, Brazil is the largest producer of coffee "
            "in the world, and over 2.25 billion cups of coffee are consumed every "
            "day globally."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────
# ⚙️ Model loading
# ─────────────────────────────────────────────────────────────────────
def load_model(model_path: str = "models/bert_model.pt",
               checkpoint: str = "distilbert-base-uncased"):
    """Load the fine-tuned DistilBERT QA model and its tokenizer."""
    print(f"\n{CYAN}{BOLD}⏳ Loading model from {model_path} ...{RESET}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(checkpoint)

    # Build model architecture from config only — no weight download needed
    config = DistilBertConfig.from_pretrained(checkpoint)
    model = DistilBertForQuestionAnswering(config)

    state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"{GREEN}{BOLD}✅ Model loaded successfully!  (device: {device}){RESET}\n")
    return model, tokenizer, device


# ─────────────────────────────────────────────────────────────────────
# 🔮 Prediction
# ─────────────────────────────────────────────────────────────────────
def predict(question: str, context: str, model, tokenizer, device) -> str:
    """Run inference and return the predicted answer string."""
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )

    offset_mapping = inputs.pop("offset_mapping")[0]
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    start_logits = outputs.start_logits[0].cpu()
    end_logits = outputs.end_logits[0].cpu()

    # Mask out non-context tokens (question + special tokens)
    # The context starts right after the first [SEP] token
    input_ids_list = inputs["input_ids"][0].tolist()
    sep_idx = input_ids_list.index(tokenizer.sep_token_id)
    mask = torch.zeros_like(start_logits, dtype=torch.bool)
    mask[sep_idx + 1:] = True

    start_logits[~mask] = -float("inf")
    end_logits[~mask] = -float("inf")

    start_idx = torch.argmax(start_logits).item()
    end_idx = torch.argmax(end_logits).item()

    if end_idx < start_idx:
        start_idx, end_idx = end_idx, start_idx

    # Map back from token positions to character positions in the context
    start_char = offset_mapping[start_idx][0].item()
    end_char = offset_mapping[end_idx][1].item()

    answer = context[start_char:end_char]
    return answer if answer.strip() else "(No answer found)"


# ─────────────────────────────────────────────────────────────────────
# 📋 Display helpers
# ─────────────────────────────────────────────────────────────────────
def show_banner():
    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════════════════╗
║              🤖 Question Answering with DistilBERT                 ║
║                    Interactive Inference CLI                        ║
╚══════════════════════════════════════════════════════════════════════╝{RESET}
""")


def show_contexts():
    """Print the numbered list of available contexts."""
    print(f"{BOLD}{YELLOW}📚 Available Contexts:{RESET}\n")
    for i, ctx in enumerate(CONTEXTS, start=1):
        # Show a short preview (first 80 chars)
        preview = ctx["text"][:80].rstrip() + "…"
        print(f"  {BOLD}{CYAN}[{i:>2}]{RESET}  {ctx['title']}")
        print(f"       {DIM}{preview}{RESET}")
    print(f"  {BOLD}{MAGENTA}[ 0]{RESET}  ✏️  Enter your own context")
    print()


def show_answer(question: str, answer: str):
    """Pretty-print the answer."""
    print(f"\n{DIVIDER}")
    print(f"  {BOLD}{YELLOW}❓ Question:{RESET}  {question}")
    print(f"  {BOLD}{GREEN}💡 Answer:{RESET}    {answer}")
    print(f"{DIVIDER}\n")


# ─────────────────────────────────────────────────────────────────────
# 🚀 Main interactive loop
# ─────────────────────────────────────────────────────────────────────
def main():
    show_banner()

    # Load config for checkpoint name
    with open("config.json") as f:
        conf = json.load(f)

    model, tokenizer, device = load_model(
        model_path=conf["model_path"],
        checkpoint=conf["checkpoint"],
    )

    while True:
        show_contexts()

        # ── Pick a context ──────────────────────────────────────────
        choice = input(f"{BOLD}👉 Select a context number (or 'q' to quit): {RESET}").strip()

        if choice.lower() in ("q", "quit", "exit"):
            print(f"\n{CYAN}👋 Goodbye!{RESET}\n")
            break

        try:
            choice = int(choice)
        except ValueError:
            print(f"{RED}⚠️  Please enter a valid number.{RESET}\n")
            continue

        if choice == 0:
            # Custom context
            print(f"\n{MAGENTA}✏️  Paste your context below (press Enter twice to finish):{RESET}")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            context = " ".join(lines).strip()
            if not context:
                print(f"{RED}⚠️  Empty context. Try again.{RESET}\n")
                continue
            title = "✏️  Custom Context"
        elif 1 <= choice <= len(CONTEXTS):
            ctx = CONTEXTS[choice - 1]
            context = ctx["text"]
            title = ctx["title"]
        else:
            print(f"{RED}⚠️  Invalid choice. Pick 1–{len(CONTEXTS)} or 0.{RESET}\n")
            continue

        # Show the selected context
        print(f"\n{DIVIDER}")
        print(f"  {BOLD}{CYAN}📖 {title}{RESET}")
        print(f"  {DIM}{context}{RESET}")
        print(f"{DIVIDER}\n")

        # ── Ask questions loop ──────────────────────────────────────
        while True:
            question = input(
                f"{BOLD}❓ Ask a question (or 'b' to go back, 'q' to quit): {RESET}"
            ).strip()

            if question.lower() in ("b", "back"):
                print()
                break
            if question.lower() in ("q", "quit", "exit"):
                print(f"\n{CYAN}👋 Goodbye!{RESET}\n")
                return
            if not question:
                continue

            answer = predict(question, context, model, tokenizer, device)
            show_answer(question, answer)


if __name__ == "__main__":
    main()
