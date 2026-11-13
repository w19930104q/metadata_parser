"""
Structured OpenGraph media extraction.

The canonical :class:`~metadata_parser.MetadataParser` storage model keeps each
``og:*`` field in a separate dict entry; repeated properties (such as several
``og:image`` roots) collapse into lists, while their ``og:image:width`` /
``og:image:height`` siblings land in *other* lists.  Re-pairing a media item
with its dimensions by list index is therefore unreliable.

This module reconstructs the structured OpenGraph media model instead:

* every root declaration (``og:image`` / ``og:video`` / ``og:audio``) opens a
  new group in document order; the ``og:<type>:url`` /
  ``og:<type>:secure_url`` alternates attach to the current group instead of
  opening another one;
* subsequent same-type sub-properties are attached to the nearest root of that
  type, so interleaved image/video/audio blocks never cross-pollinate;
* repeated root declarations are never merged, and a failure in one group is
  isolated into group-level diagnostics instead of discarding the other groups.

The extraction is read-only: the existing ``metadata`` payload is not modified,
results are cached per parser, and URL normalization reuses the document
context (final response URL, relative reconstruction and the HTTPS rules for
``secure_url`` fields) consistently with ``get_metadata_link()``.
"""

# stdlib
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import Iterator
from typing import List
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING

# pypi
from bs4 import BeautifulSoup

# local
from .regex import RE_ALL_NUMERIC
from .regex import RE_rfc3986_valid_characters
from .regex import RE_whitespace

if TYPE_CHECKING:
    from . import MetadataParser


# ==============================================================================

#: the three structured OpenGraph media families
OpenGraphMediaType = Literal["image", "video", "audio"]

#: one observed URL statement: (property, raw value, normalized URL)
MediaURLEntry = Tuple[str, str, Optional[str]]

#: one observed property statement: (property, raw value)
MediaPropertyEntry = Tuple[str, str]

#: root property names that open a new group.
#: ``og:<type>:url`` and ``og:<type>:secure_url`` are alternates of the same
#: structured object, not new roots: they attach to the nearest open group.
_ROOTS: Dict[str, "OpenGraphMediaType"] = {
    "og:image": "image",
    "og:video": "video",
    "og:audio": "audio",
}

#: sub-properties that carry a URL, grouped by media family
_URL_SUBPROPERTIES: Dict["OpenGraphMediaType", Sequence[str]] = {
    "image": ("og:image:url", "og:image:secure_url"),
    "video": ("og:video:url", "og:video:secure_url"),
    "audio": ("og:audio:url", "og:audio:secure_url"),
}

#: dimension sub-properties, grouped by media family
_DIMENSION_SUBPROPERTIES: Dict["OpenGraphMediaType", Sequence[str]] = {
    "image": ("og:image:width", "og:image:height"),
    "video": ("og:video:width", "og:video:height"),
    "audio": (),
}

#: the fields that must resolve to an HTTPS url
_FIELDS_REQUIRE_HTTPS = frozenset(
    (
        "og:image:secure_url",
        "og:video:secure_url",
        "og:audio:secure_url",
    )
)


def _media_family(property_name: str) -> Optional["OpenGraphMediaType"]:
    """Return the media family for any ``og:<family>[:...]`` property, or None.

    Recognizes both documented sub-properties and publisher-specific extras
    such as ``og:image:custom``, which still belong to the image group.
    """
    if not property_name.startswith("og:"):
        return None
    family = property_name[3:].split(":", 1)[0]
    if family in ("image", "video", "audio"):
        return family  # type: ignore[return-value]
    return None


# ==============================================================================


@dataclass(frozen=True)
class OpenGraphMediaDiagnostic:
    """A group-level problem found while assembling a media group.

    Diagnostics never abort the parse: the offending value is still retained on
    the group, and sibling groups are unaffected.

    Attributes:
        code: machine-readable problem identifier, one of

            ``"orphan_subproperty"``
                a media sub-property appeared before any root of its family
                (the value is not attached to a group);
            ``"empty_value"``
                a root or sub-property was declared with an empty value;
            ``"conflict"``
                the same single-valued sub-property received multiple distinct
                values (the first value is retained);
            ``"invalid_dimension"``
                a width/height value is not a positive integer;
            ``"invalid_url"``
                a URL value could not be normalized (the normalized URL is
                ``None`` while the raw value is retained);
            ``"secure_url_not_https"``
                a ``secure_url`` value did not resolve to HTTPS.
        media_type: the affected media family.
        property: the full OpenGraph property, e.g. ``og:image:width``.
        value: the raw value involved (may be an empty string).
        message: human-readable explanation.
    """

    code: str
    media_type: str
    property: str
    value: str
    message: str


@dataclass
class OpenGraphMediaGroup:
    """One structured media block, opened by a root declaration.

    Root declarations are never merged: two consecutive ``og:image`` tags
    produce two groups even when their contents are identical.

    Attributes:
        media_type: ``"image"``, ``"video"`` or ``"audio"``.
        index: zero-based position of this group among all media groups in
            document order (across every media family).
        root_property: the property that opened the group, always one of
            ``og:image`` / ``og:video`` / ``og:audio``.
        root_raw: the original root value as declared in the document.
        root_url: the normalized root URL, or ``None`` when normalization
            failed (see the group diagnostics).
        urls: every URL declared for the group in document order, as
            ``(property, raw_value, normalized_url)`` tuples. The first entry
            is always the root declaration.
        width / height: parsed positive-integer dimensions, or ``None``.
        media_subtype: value of the ``og:<type>:type`` sub-property, if any.
        alt: value of the ``og:<type>:alt`` sub-property, if any.
        properties: all observed sub-properties as
            ``(property, raw_value)`` tuples, in document order, including
            unknown sub-properties and values that triggered diagnostics.
        diagnostics: group-level problems; empty for a clean group.
    """

    media_type: str
    index: int
    root_property: str
    root_raw: str
    root_url: Optional[str] = None
    urls: List[MediaURLEntry] = field(default_factory=list)
    width: Optional[int] = None
    height: Optional[int] = None
    media_subtype: Optional[str] = None
    alt: Optional[str] = None
    properties: List[MediaPropertyEntry] = field(default_factory=list)
    diagnostics: List[OpenGraphMediaDiagnostic] = field(default_factory=list)

    @property
    def url(self) -> Optional[str]:
        """the normalized root URL (alias of :attr:`root_url`)."""
        return self.root_url

    @property
    def is_valid(self) -> bool:
        """``True`` when the group carries no diagnostics."""
        return not self.diagnostics


@dataclass(frozen=True)
class OpenGraphMediaQuery:
    """Result of a structured OpenGraph media query.

    Attributes:
        groups: media groups in document order, filtered by the requested
            media families when applicable.
        diagnostics: every diagnostic, including orphan sub-properties that
            could not be attached to a group, in document order.
    """

    groups: List[OpenGraphMediaGroup]
    diagnostics: List[OpenGraphMediaDiagnostic]


# ==============================================================================


def _parse_dimension(raw: str) -> Optional[int]:
    """return a positive int for `raw`, or None for anything else."""
    if raw and RE_ALL_NUMERIC.match(raw) and "." not in raw:
        # RE_ALL_NUMERIC allows a decimal point, hence the explicit guard;
        # int() also normalizes leading-zero values such as "007" -> 7
        value = int(raw)
        if value > 0:
            return value
    return None


def _normalize_media_url(
    parser: "MetadataParser",
    property_name: str,
    raw_value: str,
    allow_encoded_uri: bool,
    require_public_global: bool,
) -> Optional[str]:
    """Normalize a media URL with the same rules as `get_metadata_link()`.

    The final response URL supplies the base for relative addresses;
    ``secure_url`` fields must resolve to HTTPS.
    """
    value = RE_whitespace.sub("", raw_value)
    if not value:
        return None
    if value[:11].lower().startswith("data:image/"):
        return value if allow_encoded_uri else None
    if not RE_rfc3986_valid_characters.match(value):
        return None
    if value[0:2] == "//":
        if property_name in parser.schemeless_fields_upgradeable:
            value = parser.upgrade_schemeless_url(value, field=property_name)
    if require_public_global:
        _require_valid_netloc = True
        _allow_localhosts = False
    else:
        _require_valid_netloc = False
        _allow_localhosts = True
    # imported lazily to avoid an import cycle with the package root
    from . import is_url_valid
    from . import url_to_absolute_url

    normalized: Optional[str]
    if is_url_valid(
        value,
        require_valid_netloc=_require_valid_netloc,
        allow_localhosts=_allow_localhosts,
        urlparser=parser.urlparse,
    ):
        normalized = value
    else:
        url_fallback = parser.get_fallback_url(
            require_valid_netloc=_require_valid_netloc,
            allow_localhosts=_allow_localhosts,
        )
        normalized = url_to_absolute_url(
            value,
            url_fallback=url_fallback,
            require_valid_netloc=_require_valid_netloc,
            allow_localhosts=_allow_localhosts,
            urlparser=parser.urlparse,
        )
        if normalized is not None and not is_url_valid(
            normalized,
            require_valid_netloc=_require_valid_netloc,
            allow_localhosts=_allow_localhosts,
            urlparser=parser.urlparse,
        ):
            normalized = None
    if (
        normalized is not None
        and property_name in _FIELDS_REQUIRE_HTTPS
        and parser.urlparse(normalized).scheme != "https"
    ):
        return None
    return normalized


def _iter_og_meta_tags(
    parser: "MetadataParser",
) -> "Iterator[Tuple[str, str]]":
    """Yield ``(property, content)`` for every og meta tag in document order.

    Only available when the cached BeautifulSoup document exists: the soup
    retains the original interleaving of media roots and sub-properties.
    The search path honors ``search_head_only`` exactly like ``parse()``.
    """
    soup = parser.parsed_result.soup
    if isinstance(soup, BeautifulSoup):
        if parser.search_head_only and soup.html and soup.html.head:
            searchpath = soup.html.head
        else:
            searchpath = soup.html if soup.html else soup
        for tag in searchpath.find_all("meta"):
            if not tag.has_attr("property"):
                continue
            property_name = tag["property"]
            # matches the legacy RE_prefix_opengraph (`^og`, case-sensitive)
            if not isinstance(property_name, str) or not property_name.startswith("og"):
                continue
            content = tag.get("content", None)
            if content is None:
                # the legacy parser skips tags without a content attribute
                continue
            yield property_name, content


def _build_groups_from_flat_store(
    parser: "MetadataParser",
    allow_encoded_uri: bool,
    require_public_global: bool,
) -> "OpenGraphMediaQuery":
    """Rebuild groups from the flattened ``metadata['og']`` store.

    The flat store keeps one key per property with list values, so the exact
    interleaving of roots and sub-properties is unrecoverable. Roots are
    collected first (in store order) and sub-property values are distributed
    round-robin within each media family, which is the best reconstruction
    possible without the document and matches the common one-root-per-block
    publishing pattern.
    """
    groups: List[OpenGraphMediaGroup] = []
    diagnostics: List[OpenGraphMediaDiagnostic] = []
    # per-family cursor for round-robin sub-property distribution
    family_cursor: Dict[str, int] = {}
    family_groups: Dict[str, List[OpenGraphMediaGroup]] = {
        "image": [],
        "video": [],
        "audio": [],
    }
    group_index = 0

    def _diagnose(code, media_type, property_name, value, message, target=None):
        diagnostic = OpenGraphMediaDiagnostic(
            code=code,
            media_type=media_type,
            property=property_name,
            value=value,
            message=message,
        )
        diagnostics.append(diagnostic)
        if target is not None:
            target.diagnostics.append(diagnostic)

    og_store = parser.parsed_result.metadata.get("og", {})

    # first pass: create a group for every root value
    for family in ("image", "video", "audio"):
        root_property = "og:%s" % family
        if root_property[3:] not in og_store:
            continue
        raw = og_store[root_property[3:]]
        values = raw if isinstance(raw, list) else [raw]
        for content in values:
            stripped = content.strip()
            group = OpenGraphMediaGroup(
                media_type=family,
                index=group_index,
                root_property=root_property,
                root_raw=stripped,
            )
            group_index += 1
            groups.append(group)
            family_groups[family].append(group)
            if not stripped:
                _diagnose(
                    "empty_value",
                    family,
                    root_property,
                    stripped,
                    "root declaration `%s` has an empty value" % root_property,
                    target=group,
                )
            normalized = _normalize_media_url(
                parser,
                root_property,
                stripped,
                allow_encoded_uri=allow_encoded_uri,
                require_public_global=require_public_global,
            )
            group.root_url = normalized
            group.urls.append((root_property, stripped, normalized))
            group.properties.append((root_property, stripped))
            if stripped and normalized is None:
                _diagnose(
                    "invalid_url",
                    family,
                    root_property,
                    stripped,
                    "root URL `%s` could not be normalized" % stripped,
                    target=group,
                )

    # second pass: distribute sub-property values round-robin per family
    for key, val in og_store.items():
        property_name = "og:%s" % key
        if property_name in _ROOTS:
            continue
        sub_family = _media_family(property_name)
        if sub_family is None:
            continue
        targets = family_groups[sub_family]
        if not targets:
            for content in (val if isinstance(val, list) else [val]):
                _diagnose(
                    "orphan_subproperty",
                    sub_family,
                    property_name,
                    content.strip(),
                    "`%s` has no preceding `og:%s` root" % (property_name, sub_family),
                )
            continue
        values = val if isinstance(val, list) else [val]
        for content in values:
            cursor = family_cursor.get(sub_family, 0) % len(targets)
            family_cursor[sub_family] = cursor + 1
            target = targets[cursor]
            stripped = content.strip()
            target.properties.append((property_name, stripped))
            if not stripped:
                _diagnose(
                    "empty_value",
                    sub_family,
                    property_name,
                    stripped,
                    "sub-property `%s` has an empty value" % property_name,
                    target=target,
                )
                continue
            if property_name in _URL_SUBPROPERTIES[sub_family]:
                normalized = _normalize_media_url(
                    parser,
                    property_name,
                    stripped,
                    allow_encoded_uri=allow_encoded_uri,
                    require_public_global=require_public_global,
                )
                target.urls.append((property_name, stripped, normalized))
                if normalized is None:
                    code = (
                        "secure_url_not_https"
                        if property_name in _FIELDS_REQUIRE_HTTPS
                        else "invalid_url"
                    )
                    _diagnose(
                        code,
                        sub_family,
                        property_name,
                        stripped,
                        "URL `%s` could not be normalized" % stripped,
                        target=target,
                    )
            elif property_name in _DIMENSION_SUBPROPERTIES[sub_family]:
                dimension = _parse_dimension(stripped)
                if dimension is None:
                    _diagnose(
                        "invalid_dimension",
                        sub_family,
                        property_name,
                        stripped,
                        "`%s` must be a positive integer, got `%s`"
                        % (property_name, stripped),
                        target=target,
                    )
                    continue
                attr = "width" if property_name.endswith(":width") else "height"
                if getattr(target, attr) is None:
                    setattr(target, attr, dimension)
            elif property_name == "og:%s:type" % sub_family:
                if target.media_subtype is None:
                    target.media_subtype = stripped
            elif property_name == "og:%s:alt" % sub_family:
                if target.alt is None:
                    target.alt = stripped

    return OpenGraphMediaQuery(groups=groups, diagnostics=diagnostics)


def build_opengraph_media_groups(
    parser: "MetadataParser",
    allow_encoded_uri: bool = False,
    require_public_global: bool = True,
) -> "OpenGraphMediaQuery":
    """Assemble structured OpenGraph media groups from a parsed document.

    The function is pure with respect to the parser's metadata: it only reads
    the cached soup / ``og`` store and normalizes URLs through the parser's
    existing, side-effect-free helpers.

    ``allow_encoded_uri`` and ``require_public_global`` mirror the
    equivalently named options of ``MetadataParser.get_metadata_link()``.
    """
    if not isinstance(parser.parsed_result.soup, BeautifulSoup):
        return _build_groups_from_flat_store(
            parser,
            allow_encoded_uri=allow_encoded_uri,
            require_public_global=require_public_global,
        )

    groups: List[OpenGraphMediaGroup] = []
    diagnostics: List[OpenGraphMediaDiagnostic] = []
    # nearest open root per media family
    last_group: Dict[str, OpenGraphMediaGroup] = {}
    group_index = 0

    def _diagnose(
        code: str,
        media_type: str,
        property_name: str,
        value: str,
        message: str,
        target: Optional[OpenGraphMediaGroup] = None,
    ) -> None:
        diagnostic = OpenGraphMediaDiagnostic(
            code=code,
            media_type=media_type,
            property=property_name,
            value=value,
            message=message,
        )
        diagnostics.append(diagnostic)
        if target is not None:
            target.diagnostics.append(diagnostic)

    for property_name, content in _iter_og_meta_tags(parser):
        raw_value = content if isinstance(content, str) else str(content)
        stripped = raw_value.strip()

        # is this a root declaration?
        root_media_type = _ROOTS.get(property_name)
        if root_media_type is not None:
            group = OpenGraphMediaGroup(
                media_type=root_media_type,
                index=group_index,
                root_property=property_name,
                root_raw=stripped,
            )
            group_index += 1
            groups.append(group)
            last_group[root_media_type] = group
            if not stripped:
                _diagnose(
                    "empty_value",
                    root_media_type,
                    property_name,
                    stripped,
                    "root declaration `%s` has an empty value" % property_name,
                    target=group,
                )
            normalized = _normalize_media_url(
                parser,
                property_name,
                stripped,
                allow_encoded_uri=allow_encoded_uri,
                require_public_global=require_public_global,
            )
            group.root_url = normalized
            group.urls.append((property_name, stripped, normalized))
            group.properties.append((property_name, stripped))
            if stripped and normalized is None:
                if property_name in _FIELDS_REQUIRE_HTTPS:
                    _diagnose(
                        "secure_url_not_https",
                        root_media_type,
                        property_name,
                        stripped,
                        "`%s` must resolve to an HTTPS URL" % property_name,
                        target=group,
                    )
                else:
                    _diagnose(
                        "invalid_url",
                        root_media_type,
                        property_name,
                        stripped,
                        "root URL `%s` could not be normalized" % stripped,
                        target=group,
                    )
            continue

        # otherwise it must belong to a media family (documented or custom
        # sub-property such as `og:image:custom`); unrelated og tags are
        # ignored
        sub_media_type = _media_family(property_name)
        if sub_media_type is None:
            continue

        target = last_group.get(sub_media_type)
        if target is None:
            _diagnose(
                "orphan_subproperty",
                sub_media_type,
                property_name,
                stripped,
                "`%s` has no preceding `og:%s` root" % (property_name, sub_media_type),
            )
            continue

        target.properties.append((property_name, stripped))

        if not stripped:
            _diagnose(
                "empty_value",
                sub_media_type,
                property_name,
                stripped,
                "sub-property `%s` has an empty value" % property_name,
                target=target,
            )
            continue

        # URL sub-properties
        if property_name in _URL_SUBPROPERTIES[sub_media_type]:
            normalized = _normalize_media_url(
                parser,
                property_name,
                stripped,
                allow_encoded_uri=allow_encoded_uri,
                require_public_global=require_public_global,
            )
            target.urls.append((property_name, stripped, normalized))
            if normalized is None:
                if property_name in _FIELDS_REQUIRE_HTTPS:
                    _diagnose(
                        "secure_url_not_https",
                        sub_media_type,
                        property_name,
                        stripped,
                        "`%s` must resolve to an HTTPS URL" % property_name,
                        target=target,
                    )
                else:
                    _diagnose(
                        "invalid_url",
                        sub_media_type,
                        property_name,
                        stripped,
                        "URL `%s` could not be normalized" % stripped,
                        target=target,
                    )
            continue

        # dimensions
        if property_name in _DIMENSION_SUBPROPERTIES[sub_media_type]:
            dimension = _parse_dimension(stripped)
            if dimension is None:
                _diagnose(
                    "invalid_dimension",
                    sub_media_type,
                    property_name,
                    stripped,
                    "`%s` must be a positive integer, got `%s`"
                    % (property_name, stripped),
                    target=target,
                )
                continue
            attr = "width" if property_name.endswith(":width") else "height"
            current = getattr(target, attr)
            if current is None:
                setattr(target, attr, dimension)
            elif current != dimension:
                _diagnose(
                    "conflict",
                    sub_media_type,
                    property_name,
                    stripped,
                    "conflicting values for `%s`: retaining %s, ignoring %s"
                    % (property_name, current, dimension),
                    target=target,
                )
            continue

        # single-valued text sub-properties: type and alt
        if property_name == "og:%s:type" % sub_media_type:
            attr = "media_subtype"
        elif property_name == "og:%s:alt" % sub_media_type:
            attr = "alt"
        else:
            # publisher-specific sub-property: retained on .properties only
            continue
        current = getattr(target, attr)
        if current is None:
            setattr(target, attr, stripped)
        elif current != stripped:
            _diagnose(
                "conflict",
                sub_media_type,
                property_name,
                stripped,
                "conflicting values for `%s`: retaining `%s`, ignoring `%s`"
                % (property_name, current, stripped),
                target=target,
            )

    return OpenGraphMediaQuery(groups=groups, diagnostics=diagnostics)
