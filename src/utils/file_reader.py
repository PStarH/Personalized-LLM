import os
from typing import Optional
import docx

class FileReader:
    SUPPORTED_EXTENSIONS = ['.txt', '.docx']

    @staticmethod
    def read_file(file_path: str) -> Optional[str]:
        """
        Reads the content of a file based on its extension.

        Args:
            file_path (str): Path to the file.

        Returns:
            Optional[str]: Content of the file as a string, or None if unsupported format.
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext == '.txt':
            return FileReader._read_txt(file_path)
        elif ext == '.docx':
            return FileReader._read_docx(file_path)
        else:
            print(f"Unsupported file format for file: {file_path}")
            return None

    @staticmethod
    def _read_txt(file_path: str) -> Optional[str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading TXT file {file_path}: {e}")
            return None

    @staticmethod
    def _read_docx(file_path: str) -> Optional[str]:
        try:
            doc = docx.Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return '\n'.join(full_text)
        except Exception as e:
            print(f"Error reading DOCX file {file_path}: {e}")
            return None 