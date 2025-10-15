"""Template loading utilities for oracle generators"""

import os


class OracleTemplateLoader:
    """Loads and renders oracle templates without mutable globals"""

    def __init__(self, templates_dir: str):
        self.templates_dir = templates_dir
        self._cache = {}

    def load(self, template_file: str) -> str:
        """Load a template file and return its content"""
        if template_file in self._cache:
            return self._cache[template_file]

        template_path = os.path.join(self.templates_dir, template_file)
        with open(template_path, 'r') as f:
            content = f.read()

        self._cache[template_file] = content
        return content

    def render(self, template_file: str, **kwargs) -> str:
        """Load a template and replace {{KEY}} placeholders with values"""
        content = self.load(template_file)

        for key, value in kwargs.items():
            placeholder = f"{{{{{key}}}}}"
            content = content.replace(placeholder, str(value))

        return content


def get_oracle_template_loader() -> OracleTemplateLoader:
    """Get the oracle template loader instance"""
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    return OracleTemplateLoader(templates_dir)
