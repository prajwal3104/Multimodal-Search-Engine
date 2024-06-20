import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

__version__ = "0.0.1"

REPO_NAME = "$REPO_NAME$"
AUTHOR = "$AUTHOR$"
SRC_REPO = "$SRC_REPO$"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR,
    description="Building a dual encoder model to search for images using natural language queries",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)