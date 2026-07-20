import logging
import os
import re
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class LocalSummarizer:
    """
    Singleton-style wrapper for local LLM summarization.
    Enhances extraction using robust heuristic rules and LLM generation with retry logic.
    """
    _tokenizer = None
    _model = None
    _model_name = "facebook/bart-large-cnn"

    @classmethod
    def _load_model(cls):
        """Lazy load the model and tokenizer directly"""
        if cls._model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                import transformers
                logger.info(f"⏳ Loading summarization model ({cls._model_name})...")
                
                old_verbosity = transformers.logging.get_verbosity()
                transformers.logging.set_verbosity_error()
                
                cls._tokenizer = AutoTokenizer.from_pretrained(cls._model_name)
                cls._model = AutoModelForSeq2SeqLM.from_pretrained(cls._model_name)
                
                transformers.logging.set_verbosity(old_verbosity)
                logger.info("✅ Summarization model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load summarization model: {e}")
                cls._model = False # Mark as failed

    @staticmethod
    def _strip_yaml_frontmatter(text: str) -> str:
        """Strip the YAML frontmatter enclosed in ---"""
        return re.sub(r'^---\s*\n.*?\n---\s*\n', '', text, flags=re.MULTILINE | re.DOTALL)

    @staticmethod
    def _extract_candidates(text: str) -> List[str]:
        candidates = []
        
        # 1. Section Headers (support "1. Introduction")
        heading_matches = re.finditer(r'^#+\s*(?:\d+[\.\)]?\s*)?(Description|Model [dD]escription|Model Overview|Overview|Introduction|Summary|モデル概要|Model Details)[^\n]*\n(.*?)(?=\n#+\s|\Z)', text, flags=re.MULTILINE | re.DOTALL)
        for match in heading_matches:
            if match.group(2).strip():
                candidates.append(match.group(2).strip())
                
        # 2. Inline Labels
        inline_matches = re.finditer(r'(?:Description:|Overview:|### Description:)\s*(.*?)(?=\n\n|\Z)', text, flags=re.DOTALL | re.IGNORECASE)
        for match in inline_matches:
            if match.group(1).strip():
                candidates.append(match.group(1).strip())
                
        # 3. Auto-generated fine-tuned leading sentences
        tuned_matches = re.finditer(r'^(?:The .*model is a .*|This model is a fine-tuned version of.*|This is a fine-tuned.*)', text, flags=re.MULTILINE | re.IGNORECASE)
        for match in tuned_matches:
            candidates.append(match.group(0).strip())
            
        # 4. Fallback: First meaningful paragraph
        # Strip some HTML first just for the fallback rule
        html_stripped = re.sub(r'<[^>]+>', '', text)
        paragraphs = re.split(r'\n\s*\n', html_stripped)
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if p.startswith('#'):
                continue
            # Skip heavy markdown like links/images/badges and github alerts
            if p.startswith('[!') or p.startswith('<a href') or p.startswith('> [!'):
                continue
            # If a paragraph has many links (like a table of contents / link directory)
            if p.count('](') > 3 or p.count('http') > 3:
                continue
            if len(p) > 50:
                candidates.append(p)
                break
                
        return candidates

    @staticmethod
    def _score_candidate(text: str) -> float:
        score = 0.0
        text_lower = text.lower()
        
        # Length score (sweet spot between 100 and 500 chars)
        if 50 < len(text) < 1000:
            score += 10.0
            
        # Reward definitional patterns
        if "is a" in text_lower or "fine-tuned version of" in text_lower or "trained on" in text_lower or "designed for" in text_lower:
            score += 20.0
            
        # Penalize bad patterns
        if "leaderboard" in text_lower or "benchmark" in text_lower or "results" in text_lower:
            score -= 50.0
        if "install" in text_lower or "how to run" in text_lower or "pip install" in text_lower or "read our guide" in text_lower:
            score -= 30.0

        # Penalize quantization / repackaging marketing (e.g. unsloth/TheBloke GGUF
        # repos lead with quant claims instead of the base model's actual purpose).
        if ("outperforms other leading" in text_lower or "dynamic 2.0" in text_lower
                or "quants" in text_lower or "quantized version" in text_lower):
            score -= 40.0
            
        # Penalize table/code-heavy paragraphs and bullet points
        if text.count('|') > 5 or text.count('```') >= 1 or text.count('\n- ') > 2 or text.count('\n* ') > 2:
            score -= 50.0
            
        return score

    @staticmethod
    def _clean_text(text: str) -> str:
        # Remove HTML
        from bs4 import BeautifulSoup
        try:
            soup = BeautifulSoup(text, "html.parser")
            for tag in soup(["style", "script"]):
                tag.decompose()
            text = soup.get_text(separator=' ')
        except Exception:
            pass
            
        # Remove markdown heading markers (keep the heading text)
        text = re.sub(r'(?m)^\s{0,3}#{1,6}\s*', '', text)
        # Remove markdown images
        text = re.sub(r'!\[.*?\]\([^)]+\)', '', text)
        # Convert links to just text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # Remove markdown emphasis markers (bold/italic), keeping the inner text
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'(?<![\w*])_([^_]+)_(?![\w*])', r'\1', text)
        # Drop any stray leftover emphasis markers
        text = text.replace('**', '').replace('__', '')
        # Remove code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        # Remove inline code
        text = re.sub(r'`[^`]*`', '', text)
        # Remove tables
        text = re.sub(r'\|.*?\|', '', text)
        text = re.sub(r'(?m)^[-:| ]+$', '', text) # table separators
        
        # Remove boilerplate line by line
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            line_lower = line.lower()
            if 'generated automatically' in line_lower and 'model card' in line_lower:
                continue
            if 'completed by the model author' in line_lower:
                continue
            if 'model cards for model reporting' in line_lower:
                continue
            clean_lines.append(line)
        text = '\n'.join(clean_lines)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    @staticmethod
    def _truncate(text: str, max_output_chars: int) -> str:
        text = text.strip()
        if len(text) > max_output_chars:
            return text[:max_output_chars - 3].rstrip() + "..."
        return text

    @staticmethod
    def _strip_prompt_leak(text: str) -> str:
        """Defensively remove any of our own instruction phrasing that a
        non-instruction summarizer (BART) may echo back into its output."""
        patterns = [
            r'in one sentence,?\s*explain what this ai model is designed to do[^.]*\.?',
            r'summarize the main purpose of this ai model[^.]*\.?',
            r'based on this description:?',
        ]
        for pat in patterns:
            text = re.sub(pat, '', text, flags=re.IGNORECASE)
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def _use_ollama() -> bool:
        """Use the instruction-tuned LLM backend when AIBOM_SUMMARIZER_BACKEND=ollama."""
        return os.environ.get("AIBOM_SUMMARIZER_BACKEND", "bart").strip().lower() == "ollama"

    @classmethod
    def _generate_ollama(cls, source: str, max_output_chars: int) -> Optional[str]:
        """Generate an objective purpose statement with an instruction-tuned LLM via Ollama.

        Unlike BART, this model *follows* instructions, so we ask it directly for an
        objective, marketing-free purpose statement. Configured via env vars:
          AIBOM_OLLAMA_HOST  (default http://localhost:11434)
          AIBOM_OLLAMA_MODEL (default qwen3:4b)
        Returns None on any failure so summarize() can fall back to BART.
        """
        import json
        import urllib.request

        host = os.environ.get("AIBOM_OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        model = os.environ.get("AIBOM_OLLAMA_MODEL", "qwen3:4b")

        prompt = (
            "You are writing a factual purpose statement for an AI Bill of Materials (AIBOM).\n"
            "From the model card below, state ONLY what the model does and its intended task.\n"
            "Rules: be objective, use no marketing language, invent nothing, copy names and "
            "numbers exactly, and respond with one or two sentences only. If the purpose is "
            "not stated in the card, reply exactly: No description available.\n\n"
            f"MODEL CARD:\n{source[:6000]}"
        )
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,  # disable chain-of-thought on reasoning models (e.g. qwen3)
            "options": {"temperature": 0.0, "num_predict": 220},
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                f"{host}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.warning(f"⚠️ Ollama generation failed ({model} @ {host}): {e}")
            return None

        summary = (data.get("response") or "").strip()
        # Strip any leaked reasoning blocks from thinking models.
        summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL).strip()
        summary = cls._strip_prompt_leak(summary)

        if not summary or summary.strip().lower().rstrip('.') == "no description available":
            return None
        return cls._truncate(summary, max_output_chars)

    @classmethod
    def _generate(cls, source: str, max_output_chars: int) -> Optional[str]:
        """Summarize source text.

        NOTE: facebook/bart-large-cnn is a *summarization* model, not an
        instruction-following LLM. The ``source`` passed here MUST be the
        document text only — never an instruction/prompt. BART cannot tell an
        instruction apart from the document and will copy it into the output
        (the prompt-leak bug) and hallucinate generic "AI model designed to..."
        summaries when primed with instruction words.
        """
        if cls._model is None:
            cls._load_model()
        if not cls._model or not cls._tokenizer:
            return None

        try:
            inputs = cls._tokenizer(source, return_tensors="pt", max_length=512, truncation=True)
            generate_kwargs = {
                "max_length": 160,
                "min_length": 20,  # Avoid single word / fragment outputs
                "do_sample": False,
                "num_beams": 4,
                "early_stopping": True,
                "repetition_penalty": 2.0,
                "no_repeat_ngram_size": 3,  # Stop the repetitive looping seen on short inputs
            }
            summary_ids = cls._model.generate(inputs["input_ids"], **generate_kwargs)
            summary = cls._tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            summary = summary.strip()

            # Remove "Output:" prefix if present
            if summary.lower().startswith("output:"):
                summary = re.sub(r'^Output:\s*', '', summary, flags=re.IGNORECASE)

            summary = cls._strip_prompt_leak(summary)

            return cls._truncate(summary, max_output_chars)
        except Exception as e:
            logger.warning(f"⚠️ Generation failed: {e}")
            return None

    @staticmethod
    def _is_valid_summary(summary: str, model_id: str) -> bool:
        if not summary or len(summary) < 15:
            return False
            
        summary_lower = summary.lower()
        model_name = model_id.split('/')[-1].lower()
        
        if summary_lower == model_name or summary_lower == f"{model_name} model":
            return False
            
        # Check for markdown/html artifacts
        if '#' in summary or '<' in summary or '>' in summary or '*' in summary:
            return False
            
        # Check for instruction-like text
        if summary_lower.startswith("to install") or summary_lower.startswith("how to") or "pip install" in summary_lower:
            return False

        # Reject leaked prompt/instruction phrasing (the prompt-leak bug)
        if "explain what this ai model" in summary_lower or "summarize the main purpose" in summary_lower:
            return False
            
        # Refuse literally copying bullet points (e.g. from table)
        if "- type:" in summary_lower or "number of parameters:" in summary_lower:
            return False
            
        return True

    @classmethod
    def summarize(cls, text: str, max_output_chars: int = 1024, model_id: str = "") -> Optional[str]:
        """
        Robustly extract and summarize model description.
        """
        if not text or not text.strip():
            return None
            
        # 1. Strip YAML safely
        text_without_yaml = cls._strip_yaml_frontmatter(text)
        
        # 2. Extract multiple candidate description blocks
        candidates = cls._extract_candidates(text_without_yaml)
        
        if not candidates:
            # Fallback if candidates are absolutely empty
            candidates = [text_without_yaml[:1000]]
            
        # 3. Score candidates and pick best
        scored_candidates = [(c, cls._score_candidate(c)) for c in candidates]
        best_candidate = max(scored_candidates, key=lambda x: x[1])[0]
        
        # 4. Clean aggressively
        cleaned_text = cls._clean_text(best_candidate)
        
        if not cleaned_text.strip():
            return None

        sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)

        # Preferred backend: instruction-tuned LLM (Ollama). It follows instructions,
        # so it produces an objective purpose statement directly. Falls through to BART
        # if disabled, unavailable, or it returns an invalid summary.
        if cls._use_ollama():
            summary = cls._generate_ollama(cleaned_text, max_output_chars)
            if summary and cls._is_valid_summary(summary, model_id):
                return summary
            logger.info("⚠️ Ollama backend unavailable/invalid; falling back to BART.")

        # Use the first few sentences of the cleaned text as the source document.
        # Trailing sentences are usually training/benchmark details that dilute the
        # summary of the model's purpose.
        source = " ".join(sentences[:5]).strip()

        # BART is a summarization model and hallucinates badly on very short inputs
        # (looping repetition, invented "a new AI model designed to..." text). When we
        # don't have enough source material to summarize, return the clean extract
        # directly instead of letting the model invent a purpose.
        MIN_CHARS_FOR_ABSTRACTION = 280
        if len(source) < MIN_CHARS_FOR_ABSTRACTION:
            return cls._truncate(" ".join(sentences[:2]), max_output_chars)

        # Summarize the document text ONLY. Do not prepend an instruction/prompt:
        # BART cannot follow instructions and would echo the prompt into the output
        # (the prompt-leak bug) and produce generic hallucinations.
        summary = cls._generate(source, max_output_chars)

        if summary and cls._is_valid_summary(summary, model_id):
            return summary

        # Fallback to the cleaned extracted text (first 1-2 sentences).
        logger.info("⚠️ Summary invalid, falling back to cleaned extracted text.")
        return cls._truncate(" ".join(sentences[:2]), max_output_chars)
