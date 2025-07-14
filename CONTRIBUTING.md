# **Contributing to Hybrid Traditional-AI WAF**

We welcome and appreciate contributions from the community\! Whether it's a bug report, a feature request, a code contribution, or an improvement to the documentation, your help is valuable.

Please take a moment to review this document to understand how to contribute effectively to this project.

## **Table of Contents**

* [Code of Conduct](https://www.google.com/search?q=%23code-of-conduct)  
* [How to Contribute](https://www.google.com/search?q=%23how-to-contribute)  
  * [Reporting Bugs](https://www.google.com/search?q=%23reporting-bugs)  
  * [Suggesting Enhancements](https://www.google.com/search?q=%23suggesting-enhancements)  
  * [Pull Requests](https://www.google.com/search?q=%23pull-requests)  
* [Development Setup](https://www.google.com/search?q=%23development-setup)  
  * [Prerequisites](https://www.google.com/search?q=%23prerequisites)  
  * [Installation](https://www.google.com/search?q=%23installation)  
  * [Running the Testbed](https://www.google.com/search?q=%23running-the-testbed)  
* [Coding Guidelines](https://www.google.com/search?q=%23coding-guidelines)  
* [Commit Message Guidelines](https://www.google.com/search?q=%23commit-message-guidelines)  
* [License](https://www.google.com/search?q=%23license)

## **Code of Conduct**

This project adheres to a [Code of Conduct](https://www.google.com/search?q=CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [your.email@example.com](mailto:your.email@example.com).

## **How to Contribute**

### **Reporting Bugs**

If you find a bug, please help us by submitting an issue to our [GitHub Issues](https://www.google.com/search?q=https://github.com/your-username/your-repo-name/issues). Before submitting, please:

* **Check existing issues:** Search to see if the bug has already been reported.  
* **Provide clear details:**  
  * A clear and concise description of the bug.  
  * Steps to reproduce the behavior.  
  * Expected behavior.  
  * Actual behavior.  
  * Screenshots or error messages (if applicable).  
  * Your environment details (OS, Docker version, Python version, etc.).

### **Suggesting Enhancements**

We welcome suggestions for new features, improvements, or changes. Please submit your ideas to our [GitHub Issues](https://www.google.com/search?q=https://github.com/your-username/your-repo-name/issues) with the label enhancement. Clearly describe:

* The proposed enhancement.  
* Why it would be beneficial to the project.  
* Any alternative solutions you've considered.

### **Pull Requests**

We encourage you to contribute code by submitting pull requests\!

1. **Fork the repository:** Start by forking the Hybrid Traditional-AI WAF repository to your GitHub account.  
2. **Create a new branch:**  
   git checkout \-b feature/your-feature-name-or-bugfix/issue-number

   (e.g., feature/add-new-model/123 or bugfix/fix-logging-error/45)  
3. **Make your changes:** Implement your feature or bug fix. Ensure your code adheres to the [Coding Guidelines](https://www.google.com/search?q=%23coding-guidelines).  
4. **Test your changes:** Run relevant tests and ensure everything works as expected within the testbed. If you're adding a new feature, consider adding new tests.  
5. **Commit your changes:** Write clear and concise commit messages following the [Commit Message Guidelines](https://www.google.com/search?q=%23commit-message-guidelines).  
6. **Push your branch:**  
   git push origin feature/your-feature-name-or-bugfix/issue-number

7. **Open a Pull Request (PR):**  
   * Go to the original repository on GitHub.  
   * You should see a prompt to open a PR from your branch.  
   * Provide a clear title and description for your PR, referencing any related issues.  
   * Ensure all automated checks (if configured) pass.

## **Development Setup**

### **Prerequisites**

* [Docker](https://docs.docker.com/get-docker/) (Docker Engine and Docker Compose)  
* [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)  
* Python 3.8+ (for development, especially for ML components)

### **Installation**

1. **Clone the repository:**  
   git clone https://github.com/your-username/your-repo-name.git  
   cd your-repo-name

2. **Build and start the Docker containers:**  
   docker-compose up \--build \-d

   This command will build all necessary images and start the services in detached mode.

### **Running the Testbed**

Refer to the main [README.md](http://docs.google.com/README.md) for details on how to access and interact with the different components and operational phases of the testbed.

## **Coding Guidelines**

* **Python:** Adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/) for code style.  
* **Docstrings:** Use [Sphinx style](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html) docstrings for all functions, classes, and modules.  
* **Type Hinting:** Use Python [type hints](https://docs.python.org/3/library/typing.html) for clarity and maintainability.  
* **Comments:** Add comments where the code logic is not immediately obvious.  
* **Error Handling:** Implement robust error handling using try-except blocks.  
* **Logging:** Use the logging module for application logging, avoiding print() statements for debugging in production code.

## **Commit Message Guidelines**

We follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification for our commit messages. This leads to more readable messages and makes it easier to generate changelogs.

Examples:

* feat: Add new XSS attack payload generation  
* fix: Correct SQLi payload encoding issue  
* docs: Update Dataset Generation documentation  
* chore: Upgrade Docker Compose version  
* refactor: Improve WAF log parsing logic

## **License**

By contributing to Hybrid Traditional-AI WAF, you agree that your contributions will be licensed under the project's [MIT License](https://www.google.com/search?q=LICENSE).

## **Contact**

For any questions or further assistance, please open an issue or contact [your.email@example.com](mailto:your.email@example.com).