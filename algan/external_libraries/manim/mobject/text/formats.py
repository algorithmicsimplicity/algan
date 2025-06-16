from collections.abc import Iterable, Sequence

from algan.external_libraries.manim import Tex
from algan.external_libraries.manim.mobject.types.vectorized_mobject import VGroup, VMobject
import numpy as np


class Paragraph(VGroup):
    r"""Display a paragraph of text.

    For a given :class:`.Paragraph` ``par``, the attribute ``par.chars`` is a
    :class:`.VGroup` containing all the lines. In this context, every line is
    constructed as a :class:`.VGroup` of characters contained in the line.


    Parameters
    ----------
    line_spacing
        Represents the spacing between lines. Defaults to -1, which means auto.
    alignment
        Defines the alignment of paragraph. Defaults to None. Possible values are "left", "right" or "center".

    Examples
    --------
    Normal usage::

        paragraph = Paragraph(
            "this is a awesome",
            "paragraph",
            "With \nNewlines",
            "\tWith Tabs",
            "  With Spaces",
            "With Alignments",
            "center",
            "left",
            "right",
        )

    Remove unwanted invisible characters::

        self.play(Transform(remove_invisible_chars(paragraph.chars[0:2]),
                            remove_invisible_chars(paragraph.chars[3][0:3]))

    """

    def __init__(
        self,
        *text: Sequence[str],
        line_spacing: float = -1,
        alignment: str | None = None,
        **kwargs,
    ) -> None:
        self.line_spacing = line_spacing
        self.alignment = alignment
        self.consider_spaces_as_chars = kwargs.get("disable_ligatures", False)
        super().__init__()

        lines_str = "\n".join([f'${_}$' for _ in list(text)])
        self.lines_text = Tex(lines_str, line_spacing=line_spacing, **kwargs)
        lines_str_list = lines_str.split("\n")
        self.chars = self._gen_chars(lines_str_list)

        self.lines = [list(self.chars), [self.alignment] * len(self.chars)]
        self.lines_initial_positions = [line.get_center() for line in self.lines[0]]
        self.add(*self.lines[0])
        self.move_to(np.array([0, 0, 0]))
        if self.alignment:
            self._set_all_lines_alignments(self.alignment)

    def _gen_chars(self, lines_str_list: list) -> VGroup:
        """Function to convert a list of plain strings to a VGroup of VGroups of chars.

        Parameters
        ----------
        lines_str_list
            List of plain text strings.

        Returns
        -------
        :class:`~.VGroup`
            The generated 2d-VGroup of chars.
        """
        char_index_counter = 0
        chars = self.get_group_class()()
        for line_no in range(len(lines_str_list)):
            line_str = lines_str_list[line_no]
            # Count all the characters in line_str
            # Spaces may or may not count as characters
            if self.consider_spaces_as_chars:
                char_count = len(line_str)
            else:
                char_count = 0
                for char in line_str:
                    if not char.isspace():
                        char_count += 1

            chars.add(self.get_group_class()())
            chars[line_no].add(
                *self.lines_text.chars[
                    char_index_counter : char_index_counter + char_count
                ]
            )
            char_index_counter += char_count
            if self.consider_spaces_as_chars:
                # If spaces count as characters, count the extra \n character
                # which separates Paragraph's lines to avoid issues
                char_index_counter += 1
        return chars

    def _set_all_lines_alignments(self, alignment: str):
        """Function to set all line's alignment to a specific value.

        Parameters
        ----------
        alignment
            Defines the alignment of paragraph. Possible values are "left", "right", "center".
        """
        for line_no in range(len(self.lines[0])):
            self._change_alignment_for_a_line(alignment, line_no)
        return self

    def _set_line_alignment(self, alignment: str, line_no: int):
        """Function to set one line's alignment to a specific value.

        Parameters
        ----------
        alignment
            Defines the alignment of paragraph. Possible values are "left", "right", "center".
        line_no
            Defines the line number for which we want to set given alignment.
        """
        self._change_alignment_for_a_line(alignment, line_no)
        return self

    def _set_all_lines_to_initial_positions(self):
        """Set all lines to their initial positions."""
        self.lines[1] = [None] * len(self.lines[0])
        for line_no in range(len(self.lines[0])):
            self[line_no].move_to(
                self.get_center() + self.lines_initial_positions[line_no],
            )
        return self

    def _set_line_to_initial_position(self, line_no: int):
        """Function to set one line to initial positions.

        Parameters
        ----------
        line_no
            Defines the line number for which we want to set given alignment.
        """
        self.lines[1][line_no] = None
        self[line_no].move_to(self.get_center() + self.lines_initial_positions[line_no])
        return self

    def _change_alignment_for_a_line(self, alignment: str, line_no: int) -> None:
        """Function to change one line's alignment to a specific value.

        Parameters
        ----------
        alignment
            Defines the alignment of paragraph. Possible values are "left", "right", "center".
        line_no
            Defines the line number for which we want to set given alignment.
        """
        self.lines[1][line_no] = alignment
        if self.lines[1][line_no] == "center":
            self[line_no].move_to(
                np.array([self.get_center()[0], self[line_no].get_center()[1], 0]),
            )
        elif self.lines[1][line_no] == "right":
            self[line_no].move_to(
                np.array(
                    [
                        self.get_right()[0] - self[line_no].width / 2,
                        self[line_no].get_center()[1],
                        0,
                    ],
                ),
            )
        elif self.lines[1][line_no] == "left":
            self[line_no].move_to(
                np.array(
                    [
                        self.get_left()[0] + self[line_no].width / 2,
                        self[line_no].get_center()[1],
                        0,
                    ],
                ),
            )
