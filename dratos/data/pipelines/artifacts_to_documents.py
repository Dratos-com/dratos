import gzip
import mimetypes
from typing import List, Optional, Callable
import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab
from daft import col
from dratos.data.obj.artifacts.artifact_obj import Artifact


class SegmentationStrategy:
    """
    Enumeration of available segmentation strategies.
    """

    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    CUSTOM = "custom"


class ArtifactsToSpacyPipeline:
    """
    spaCy pipeline for converting Artifact DataFrames into spaCy Documents with annotations and vocabulary.
    Supports multiple segmentation strategies.
    """

    def __init__(
        self,
        artifact: Artifact,
        mimetypes: Optional[List[str]] = None,
        nlp_model: str = "en_core_web_sm",
        segmentation_strategy: str = SegmentationStrategy.SENTENCE,
        custom_segmenter: Optional[Callable[[str], List[str]]] = None,
    ):
        """
        Initializes the pipeline.

        Parameters:
            artifact (Artifact): The Artifact instance containing the DataFrame of artifacts.
            mimetypes (Optional[List[str]]): List of MIME types to process. Defaults to text/ types.
            nlp_model (str): The spaCy model to use. Defaults to 'en_core_web_sm'.
            segmentation_strategy (str): The segmentation strategy to use ('sentence', 'paragraph', or 'custom').
            custom_segmenter (Optional[Callable[[str], List[str]]]): Custom segmentation function if strategy is 'custom'.
        """
        self.artifact = artifact
        self.mimetypes = mimetypes or [
            "text/plain",
            "application/json",
            "application/xml",
        ]
        self.nlp = spacy.load(nlp_model)
        self.vocab = self.nlp.vocab
        self.docs = []
        self.segmentation_strategy = segmentation_strategy
        self.custom_segmenter = custom_segmenter

        # Validate segmentation strategy
        if self.segmentation_strategy not in [
            SegmentationStrategy.SENTENCE,
            SegmentationStrategy.PARAGRAPH,
            SegmentationStrategy.CUSTOM,
        ]:
            raise ValueError(
                f"Unsupported segmentation strategy: {self.segmentation_strategy}"
            )
        if (
            self.segmentation_strategy == SegmentationStrategy.CUSTOM
            and not self.custom_segmenter
        ):
            raise ValueError(
                "Custom segmenter function must be provided for 'custom' strategy."
            )

    def filter_artifacts(self) -> None:
        """
        Filters the artifacts DataFrame to include only the specified MIME types.
        """
        self.filtered_df = self.artifact.df.where(col("mime_type").isin(self.mimetypes))

    def extract_text(self, payload: bytes) -> str:
        """
        Decompresses and decodes the payload to extract text.

        Parameters:
            payload (bytes): The gzipped content of the artifact.

        Returns:
            str: The extracted text content.
        """
        decompressed = gzip.decompress(payload)
        return decompressed.decode("utf-8")

    def segment_text(self, text: str) -> List[str]:
        """
        Segments text based on the selected segmentation strategy.

        Parameters:
            text (str): The raw text to segment.

        Returns:
            List[str]: A list of text segments.
        """
        if self.segmentation_strategy == SegmentationStrategy.SENTENCE:
            doc = self.nlp(text)
            segments = [sent.text.strip() for sent in doc.sents]
            return segments
        elif self.segmentation_strategy == SegmentationStrategy.PARAGRAPH:
            segments = [para.strip() for para in text.split("\n\n") if para.strip()]
            return segments
        elif self.segmentation_strategy == SegmentationStrategy.CUSTOM:
            return self.custom_segmenter(text)
        else:
            # Fallback to entire text as a single segment
            return [text]

    def create_docs(self) -> None:
        """
        Converts filtered artifacts into spaCy Doc objects using the selected segmentation strategy.
        """
        for artifact_record in self.filtered_df.collect().to_pydict():
            text = self.extract_text(artifact_record["payload"])
            segments = self.segment_text(text)
            for segment in segments:
                doc = self.nlp(segment)
                # Add custom annotations if needed
                doc.user_data["artifact_id"] = artifact_record["id"]
                self.docs.append(doc)

    def build_vocab(self) -> Vocab:
        """
        Builds the spaCy vocabulary based on the created documents.

        Returns:
            Vocab: The built spaCy vocabulary.
        """
        for doc in self.docs:
            for word in doc:
                _ = self.vocab[word.text]
        return self.vocab

    def run_pipeline(self) -> List[Doc]:
        """
        Executes the entire pipeline: filtering, converting to docs, and building vocab.

        Returns:
            List[Doc]: A list of spaCy Doc objects with annotations.
        """
        self.filter_artifacts()
        self.create_docs()
        self.build_vocab()
        return self.docs

    def add_annotations(self, doc: Doc) -> None:
        """
        Adds custom annotations to a spaCy Doc.

        Parameters:
            doc (Doc): The spaCy Doc to annotate.
        """
        # Example: Adding named entities (dummy implementation)
        # Replace with actual annotation logic as needed
        with doc.retokenize() as retokenizer:
            for token in doc:
                if token.text.lower() == "spacy":
                    retokenizer.merge(
                        token, attrs={"ENT_TYPE": "ORG", "LEMMA": "spacy"}
                    )
        # Alternatively, use custom components or external annotation tools


# Usage Example
if __name__ == "__main__":
    # Initialize Artifact instance with desired files
    artifact = Artifact(files=["path/to/your/textfile.txt"])

    # Define a custom segmentation function (optional)
    def custom_segmenter(text: str) -> List[str]:
        # Example: Split text by a specific delimiter
        return [
            segment.strip() for segment in text.split("===END===") if segment.strip()
        ]

    # Initialize the pipeline with different segmentation strategies
    # 1. Sentence Segmentation
    sentence_pipeline = ArtifactsToSpacyPipeline(
        artifact=artifact, segmentation_strategy=SegmentationStrategy.SENTENCE
    )
    sentence_docs = sentence_pipeline.run_pipeline()
    for doc in sentence_docs:
        sentence_pipeline.add_annotations(doc)
        print("Sentence Segment:", doc.text)

    # 2. Paragraph Segmentation
    paragraph_pipeline = ArtifactsToSpacyPipeline(
        artifact=artifact, segmentation_strategy=SegmentationStrategy.PARAGRAPH
    )
    paragraph_docs = paragraph_pipeline.run_pipeline()
    for doc in paragraph_docs:
        paragraph_pipeline.add_annotations(doc)
        print("Paragraph Segment:", doc.text)

    # 3. Custom Segmentation
    custom_pipeline = ArtifactsToSpacyPipeline(
        artifact=artifact,
        segmentation_strategy=SegmentationStrategy.CUSTOM,
        custom_segmenter=custom_segmenter,
    )
    custom_docs = custom_pipeline.run_pipeline()
    for doc in custom_docs:
        custom_pipeline.add_annotations(doc)
        print("Custom Segment:", doc.text)
