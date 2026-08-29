"""Public entry point for revised-paper Table 9 feature taxonomy generation."""
from generate_table10 import inventory, main, write_csv, write_latex, write_markdown


__all__ = ["inventory", "main", "write_csv", "write_latex", "write_markdown"]


if __name__ == "__main__":
    raise SystemExit(main())
