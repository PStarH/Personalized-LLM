import os
from typing import Optional
import docx
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
import pandas as pd
from bs4 import BeautifulSoup
import json
import xml.etree.ElementTree as ET
import yaml
from striprtf.striprtf import rtf_to_text
import configparser

class FileReader:
    SUPPORTED_EXTENSIONS = ['.txt', '.docx', '.pdf', '.xlsx', '.csv', '.html', '.md', '.json', '.xml', '.yaml', '.yml', '.rtf', '.ini']

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
        elif ext == '.pdf':
            return FileReader._read_pdf(file_path)
        elif ext == '.xlsx':
            return FileReader._read_xlsx(file_path)
        elif ext == '.csv':
            return FileReader._read_csv(file_path)
        elif ext == '.html':
            return FileReader._read_html(file_path)
        elif ext == '.md':
            return FileReader._read_md(file_path)
        elif ext == '.json':
            return FileReader._read_json(file_path)
        elif ext == '.xml':
            return FileReader._read_xml(file_path)
        elif ext in ['.yaml', '.yml']:
            return FileReader._read_yaml(file_path)
        elif ext == '.rtf':
            return FileReader._read_rtf(file_path)
        elif ext == '.ini':
            return FileReader._read_ini(file_path)
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

    @staticmethod
    def _read_pdf(file_path: str) -> Optional[str]:
        try:
            text = ""
            # Attempt to extract text using PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text += extracted_text
            if text.strip():
                return text
            else:
                # If no text extracted, perform OCR
                images = convert_from_path(file_path)
                for image in images:
                    text += pytesseract.image_to_string(image)
                return text
        except Exception as e:
            print(f"Error reading PDF file {file_path}: {e}")
            return None

    @staticmethod
    def _read_xlsx(file_path: str) -> Optional[str]:
        try:
            df = pd.read_excel(file_path)
            return df.to_string(index=False)
        except Exception as e:
            print(f"Error reading XLSX file {file_path}: {e}")
            return None

    @staticmethod
    def _read_csv(file_path: str) -> Optional[str]:
        try:
            df = pd.read_csv(file_path)
            return df.to_string(index=False)
        except Exception as e:
            print(f"Error reading CSV file {file_path}: {e}")
            return None

    @staticmethod
    def _read_html(file_path: str) -> Optional[str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                return soup.get_text()
        except Exception as e:
            print(f"Error reading HTML file {file_path}: {e}")
            return None

    @staticmethod
    def _read_md(file_path: str) -> Optional[str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading Markdown file {file_path}: {e}")
            return None

    @staticmethod
    def _read_json(file_path: str) -> Optional[str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return json.dumps(data, indent=4)
        except Exception as e:
            print(f"Error reading JSON file {file_path}: {e}")
            return None

    @staticmethod
    def _read_xml(file_path: str) -> Optional[str]:
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            return ET.tostring(root, encoding='utf-8').decode('utf-8')
        except Exception as e:
            print(f"Error reading XML file {file_path}: {e}")
            return None

    @staticmethod
    def _read_yaml(file_path: str) -> Optional[str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return yaml.dump(data, sort_keys=False)
        except Exception as e:
            print(f"Error reading YAML file {file_path}: {e}")
            return None

    @staticmethod
    def _read_rtf(file_path: str) -> Optional[str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                rtf_content = f.read()
                return rtf_to_text(rtf_content)
        except Exception as e:
            print(f"Error reading RTF file {file_path}: {e}")
            return None

    @staticmethod
    def _read_ini(file_path: str) -> Optional[str]:
        try:
            config = configparser.ConfigParser()
            config.read(file_path, encoding='utf-8')
            return json.dumps({section: dict(config.items(section)) for section in config.sections()}, indent=4)
        except Exception as e:
            print(f"Error reading INI file {file_path}: {e}")
            return None