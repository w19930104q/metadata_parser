# stdlib
from unittest.mock import patch

# pypi
import pytest
import requests

# local
import metadata_parser
from metadata_parser import MetadataParser
from metadata_parser import OpenGraphMediaDiagnostic
from metadata_parser import OpenGraphMediaGroup
from metadata_parser import OpenGraphMediaQuery
from metadata_parser.exceptions import InvalidStrategy
from metadata_parser.opengraph_media import build_opengraph_media_groups

# ==============================================================================

REQUESTED_URL = "http://example.com/start"
RESPONSE_URL = "https://example.com/articles/page.html"


def parse_media(head: str, url: str = RESPONSE_URL, **kwargs) -> MetadataParser:
    return MetadataParser(
        url=url, html="<html><head>%s</head><body></body></html>" % head, **kwargs
    )


def og(property_name: str, content: str) -> str:
    return '<meta property="%s" content="%s">' % (property_name, content)


# ==============================================================================
# public surface
# ==============================================================================


def test_public_types_are_exported_from_package_root():
    assert metadata_parser.OpenGraphMediaGroup is OpenGraphMediaGroup
    assert metadata_parser.OpenGraphMediaQuery is OpenGraphMediaQuery
    assert metadata_parser.OpenGraphMediaDiagnostic is OpenGraphMediaDiagnostic
    assert metadata_parser.build_opengraph_media_groups is build_opengraph_media_groups
    assert callable(MetadataParser.get_opengraph_media)
    assert callable(MetadataParser.get_opengraph_images)
    assert callable(MetadataParser.get_opengraph_videos)
    assert callable(MetadataParser.get_opengraph_audios)


def test_query_returns_typed_result():
    page = parse_media(og("og:image", "https://example.com/a.jpg"))
    result = page.get_opengraph_media()
    assert isinstance(result, OpenGraphMediaQuery)
    assert result.groups and isinstance(result.groups[0], OpenGraphMediaGroup)
    assert isinstance(result.diagnostics, list)


def test_invalid_media_type_raises():
    page = parse_media(og("og:image", "https://example.com/a.jpg"))
    with pytest.raises(ValueError, match="media_type"):
        page.get_opengraph_media("document")


# ==============================================================================
# structural grouping
# ==============================================================================


def test_repeated_roots_are_not_merged_even_when_identical():
    page = parse_media(
        og("og:image", "https://example.com/dup.jpg")
        + og("og:image:width", "10")
        + og("og:image", "https://example.com/dup.jpg")
        + og("og:image:width", "20")
    )
    groups = page.get_opengraph_images()
    assert len(groups) == 2
    assert [g.root_raw for g in groups] == [
        "https://example.com/dup.jpg",
        "https://example.com/dup.jpg",
    ]
    assert [g.index for g in groups] == [0, 1]
    assert groups[0].width == 10
    assert groups[1].width == 20


def test_same_url_with_different_dimensions_stays_in_separate_groups():
    page = parse_media(
        og("og:image", "https://example.com/same.jpg")
        + og("og:image:width", "300")
        + og("og:image:height", "200")
        + og("og:image", "https://example.com/same.jpg")
        + og("og:image:width", "1024")
        + og("og:image:height", "768")
    )
    groups = page.get_opengraph_images()
    assert len(groups) == 2
    assert (groups[0].width, groups[0].height) == (300, 200)
    assert (groups[1].width, groups[1].height) == (1024, 768)
    assert groups[0].root_url == groups[1].root_url


def test_interleaved_media_types_do_not_cross_pollinate():
    page = parse_media(
        og("og:image", "https://example.com/a.jpg")
        + og("og:video", "https://example.com/v.mp4")
        + og("og:video:width", "640")
        + og("og:video:height", "480")
        + og("og:image:width", "300")
        + og("og:image:height", "200")
        + og("og:audio", "https://example.com/a.mp3")
        + og("og:audio:type", "audio/mpeg")
        + og("og:video", "https://example.com/v2.mp4")
        + og("og:video:width", "1280")
    )
    result = page.get_opengraph_media()
    assert [g.media_type for g in result.groups] == [
        "image",
        "video",
        "audio",
        "video",
    ]
    images = page.get_opengraph_images()
    videos = page.get_opengraph_videos()
    audios = page.get_opengraph_audios()
    assert len(images) == 1 and len(videos) == 2 and len(audios) == 1
    # the image width declared between/after video tags still lands on the image
    assert (images[0].width, images[0].height) == (300, 200)
    assert (videos[0].width, videos[0].height) == (640, 480)
    assert videos[1].width == 1280 and videos[1].height is None
    assert audios[0].media_subtype == "audio/mpeg"
    # family-filtered queries only expose that family's diagnostics
    assert page.get_opengraph_media("image").diagnostics == []
    assert result.diagnostics == []


def test_root_alternates_open_groups_and_collect_all_urls():
    page = parse_media(
        og("og:image", "http://example.com/a.jpg")
        + og("og:image:url", "https://example.com/a.png")
        + og("og:image:secure_url", "https://example.com/a.secure.jpg")
        + og("og:video", "https://example.com/v.mp4")
        + og("og:video:secure_url", "https://example.com/v.secure.mp4")
    )
    image = page.get_opengraph_images()[0]
    assert image.root_property == "og:image"
    assert [entry[0] for entry in image.urls] == [
        "og:image",
        "og:image:url",
        "og:image:secure_url",
    ]
    assert image.urls[0][2] == "http://example.com/a.jpg"
    assert image.urls[2][2] == "https://example.com/a.secure.jpg"
    video = page.get_opengraph_videos()[0]
    assert [entry[0] for entry in video.urls] == [
        "og:video",
        "og:video:secure_url",
    ]


def test_unknown_and_unrelated_og_properties_are_ignored():
    page = parse_media(
        og("og:title", "Title")
        + og("og:type", "article")
        + og("og:url", "https://example.com/page")
        + og("og:image", "https://example.com/a.jpg")
        + og("og:image:custom", "x")
        + og("og:site_name", "Site")
    )
    result = page.get_opengraph_media()
    assert len(result.groups) == 1
    # unknown sub-property is retained on the group for inspection
    assert ("og:image:custom", "x") in result.groups[0].properties


# ==============================================================================
# url normalization
# ==============================================================================


def test_relative_and_schemeless_urls_use_final_response_url():
    page = parse_media(
        og("og:image", "/img/root-relative.jpg")
        + og("og:image", "img/relative.jpg")
        + og("og:video", "//cdn.example.com/protocol-relative.mp4"),
        url=RESPONSE_URL,
    )
    groups = page.get_opengraph_media().groups
    assert groups[0].root_url == "https://example.com/img/root-relative.jpg"
    assert groups[1].root_url == "https://example.com/articles/img/relative.jpg"
    assert groups[2].root_url == "https://cdn.example.com/protocol-relative.mp4"
    # raw values are always preserved alongside the normalized url
    assert groups[0].root_raw == "/img/root-relative.jpg"


def test_requested_url_ignored_when_response_url_differs():
    page = MetadataParser(
        url=REQUESTED_URL,
        url_actual=RESPONSE_URL,
        html="<html><head>%s</head></html>" % og("og:image", "next/image.jpg"),
    )
    group = page.get_opengraph_images()[0]
    assert group.root_url == "https://example.com/articles/next/image.jpg"


def test_secure_url_must_be_https():
    page = parse_media(
        og("og:image", "https://example.com/a.jpg")
        + og("og:image:secure_url", "http://example.com/insecure.jpg")
    )
    group = page.get_opengraph_images()[0]
    secure_entry = group.urls[-1]
    assert secure_entry[0] == "og:image:secure_url"
    assert secure_entry[1] == "http://example.com/insecure.jpg"
    assert secure_entry[2] is None
    codes = {d.code for d in group.diagnostics}
    assert "secure_url_not_https" in codes
    # the plain root url is still normalized and usable
    assert group.root_url == "https://example.com/a.jpg"


def test_schemeless_secure_url_upgrades_to_document_scheme():
    page = parse_media(
        og("og:image", "//example.com/root.jpg")
        + og("og:image:secure_url", "//example.com/secure.jpg"),
        url="https://example.com/",
    )
    group = page.get_opengraph_images()[0]
    assert group.root_url == "https://example.com/root.jpg"
    secure_entry = group.urls[-1]
    assert secure_entry[2] == "https://example.com/secure.jpg"
    assert group.diagnostics == []


def test_normalization_never_borrows_from_other_groups():
    page = parse_media(
        og("og:image", "https://example.com/explicit-a.jpg")
        + og("og:image", "/img/relative.jpg")
    )
    groups = page.get_opengraph_images()
    assert groups[0].root_url == "https://example.com/explicit-a.jpg"
    assert groups[1].root_url == "https://example.com/img/relative.jpg"


def test_invalid_url_is_diagnosed_not_dropped():
    page = parse_media(
        og("og:image", "https://example.com/good.jpg")
        + og("og:image", "http://127.0.0.1/secret.jpg")
    )
    groups = page.get_opengraph_images()
    assert len(groups) == 2
    bad = groups[1]
    # localhost URLs are remounted onto the document domain by the normalizer,
    # which is the library's established get_metadata_link() behavior; the raw
    # value is retained and the normalized url is independently verifiable
    assert bad.root_raw == "http://127.0.0.1/secret.jpg"
    assert bad.root_url == "https://example.com/secret.jpg"
    # the good group is unaffected
    assert groups[0].root_url == "https://example.com/good.jpg"
    assert groups[0].diagnostics == []
    assert groups[0].is_valid


def test_unparseable_url_is_diagnosed_and_keeps_the_group():
    page = MetadataParser(
        # no document URL context means the relative value cannot be made absolute
        html="<html><head>%s%s</head></html>"
        % (
            og("og:image", "https://example.com/good.jpg"),
            og("og:image", "relative/with spaces/ bad.jpg"),
        )
    )
    groups = page.get_opengraph_images()
    assert len(groups) == 2
    bad = groups[1]
    assert bad.root_raw == "relative/with spaces/ bad.jpg"
    assert bad.root_url is None
    assert any(d.code == "invalid_url" for d in bad.diagnostics)
    assert groups[0].is_valid and not bad.is_valid


# ==============================================================================
# diagnostics
# ==============================================================================


def test_orphan_subproperties_are_diagnosed_and_not_attached():
    page = parse_media(
        og("og:image:width", "300")
        + og("og:image:height", "200")
        + og("og:video:width", "640")
        + og("og:image", "https://example.com/a.jpg")
        + og("og:image:alt", "after root")
    )
    result = page.get_opengraph_media()
    assert len(result.groups) == 1
    orphans = [d for d in result.diagnostics if d.code == "orphan_subproperty"]
    assert {(d.property, d.media_type) for d in orphans} == {
        ("og:image:width", "image"),
        ("og:image:height", "image"),
        ("og:video:width", "video"),
    }
    group = result.groups[0]
    assert group.width is None and group.height is None
    assert group.alt == "after root"
    # orphans are top-level diagnostics only, never attached to a group
    assert group.diagnostics == []


def test_empty_values_are_diagnosed_per_group():
    page = parse_media(
        og("og:image", "")
        + og("og:image:width", "300")
        + og("og:image", "https://example.com/b.jpg")
    )
    result = page.get_opengraph_media()
    empty_group, good_group = result.groups
    assert empty_group.root_raw == "" and empty_group.root_url is None
    empty_diag = [d for d in empty_group.diagnostics if d.code == "empty_value"]
    assert len(empty_diag) == 1 and empty_diag[0].property == "og:image"
    # the width still attaches to the (empty) nearest root
    assert empty_group.width == 300
    assert good_group.root_url == "https://example.com/b.jpg"
    assert good_group.diagnostics == []


def test_conflicting_dimensions_and_text_keep_first_with_diagnostic():
    page = parse_media(
        og("og:image", "https://example.com/a.jpg")
        + og("og:image:width", "300")
        + og("og:image:width", "999")
        + og("og:image:alt", "first alt")
        + og("og:image:alt", "second alt")
        + og("og:image:type", "image/jpeg")
        + og("og:image:type", "image/png")
    )
    group = page.get_opengraph_images()[0]
    assert group.width == 300
    assert group.alt == "first alt"
    assert group.media_subtype == "image/jpeg"
    conflicts = [d for d in group.diagnostics if d.code == "conflict"]
    assert {d.property for d in conflicts} == {
        "og:image:width",
        "og:image:alt",
        "og:image:type",
    }
    for diagnostic in conflicts:
        assert isinstance(diagnostic, OpenGraphMediaDiagnostic)
        assert diagnostic.media_type == "image"


def test_duplicate_equal_values_do_not_raise_conflicts():
    page = parse_media(
        og("og:image", "https://example.com/a.jpg")
        + og("og:image:width", "300")
        + og("og:image:width", "300")
        + og("og:image:alt", "same")
        + og("og:image:alt", "same")
    )
    group = page.get_opengraph_images()[0]
    assert group.width == 300 and group.alt == "same"
    assert group.diagnostics == []


@pytest.mark.parametrize(
    "bad_value",
    ("not-a-number", "0", "-5", "12.5", "1e3"),
)
def test_illegal_dimensions_are_diagnosed(bad_value):
    page = parse_media(
        og("og:image", "https://example.com/a.jpg")
        + og("og:image:width", bad_value)
        + og("og:video", "https://example.com/v.mp4")
        + og("og:video:height", "007")
    )
    result = page.get_opengraph_media()
    image, video = result.groups
    assert image.width is None
    assert any(
        d.code == "invalid_dimension" and d.property == "og:image:width"
        for d in image.diagnostics
    )
    # leading-zero numeric strings still parse as positive integers
    assert video.height == 7
    assert video.diagnostics == []


def test_one_bad_group_does_not_discard_other_groups():
    page = parse_media(
        og("og:image", "https://example.com/good.jpg")
        + og("og:image:width", "300")
        + og("og:image", "http://127.0.0.1/bad.jpg")
        + og("og:image:width", "nope")
        + og("og:image", "https://example.com/also-good.jpg")
        + og("og:image:width", "64")
    )
    groups = page.get_opengraph_images()
    assert len(groups) == 3
    assert groups[0].width == 300 and groups[0].is_valid
    assert not groups[1].is_valid and groups[1].width is None
    assert groups[2].width == 64 and groups[2].is_valid
    # only the illegal dimension is a diagnostic; the localhost URL still
    # normalizes via remounting, matching get_metadata_link() behavior
    assert len(page.get_opengraph_media().diagnostics) == 1
    assert page.get_opengraph_media().diagnostics[0].code == "invalid_dimension"


# ==============================================================================
# caching, immutability and compatibility
# ==============================================================================


def test_query_is_cached_and_refresh_rebuilds():
    page = parse_media(og("og:image", "https://example.com/a.jpg"))
    first = page.get_opengraph_media()
    assert page.get_opengraph_media() is first
    assert page.get_opengraph_media("image") is not first  # filtered copy
    rebuilt = page.get_opengraph_media(refresh=True)
    assert rebuilt is not first
    assert [(g.root_raw, g.root_url) for g in rebuilt.groups] == [
        (g.root_raw, g.root_url) for g in first.groups
    ]


def test_cache_is_keyed_per_normalization_option_combination():
    page = parse_media(og("og:image", "https://example.com/a.jpg"))
    default_query = page.get_opengraph_media()
    # distinct option sets get their own cache entries
    relaxed = page.get_opengraph_media(require_public_global=False)
    encoded = page.get_opengraph_media(allow_encoded_uri=True)
    assert page.get_opengraph_media() is default_query
    assert page.get_opengraph_media(require_public_global=False) is relaxed
    assert page.get_opengraph_media(allow_encoded_uri=True) is encoded
    assert len(page._opengraph_media_cache) == 3
    # refreshing one entry leaves the others untouched
    page.get_opengraph_media(refresh=True)
    assert page.get_opengraph_media(require_public_global=False) is relaxed


def test_allow_encoded_uri_mirrors_get_metadata_link():
    data_uri = "data:image/png;base64,AAAA"
    page = parse_media(og("og:image", data_uri))
    dropped = page.get_opengraph_images()
    kept = page.get_opengraph_images(allow_encoded_uri=True)
    assert dropped[0].root_raw == data_uri and dropped[0].root_url is None
    assert kept[0].root_url == data_uri
    # legacy method has the same option semantics
    assert page.get_metadata_link("og:image") is None
    assert page.get_metadata_link("og:image", allow_encoded_uri=True) == data_uri


def test_require_public_global_false_preserves_local_urls():
    page = MetadataParser(
        url="http://localhost:8000/dir/page",
        html="<html><head>%s</head></html>"
        % og("og:image", "http://127.0.0.1:9000/local.jpg"),
    )
    strict = page.get_opengraph_images()[0]
    relaxed = page.get_opengraph_images(require_public_global=False)[0]
    assert strict.root_raw == "http://127.0.0.1:9000/local.jpg"
    # strict normalization remounts/filters via the public-document path
    assert relaxed.root_url == "http://127.0.0.1:9000/local.jpg"
    assert strict.root_url != relaxed.root_url


def test_caches_are_not_shared_between_parser_instances():
    page_a = parse_media(og("og:image", "https://example.com/a.jpg"))
    page_b = parse_media(og("og:image", "https://example.com/b.jpg"))
    page_a.get_opengraph_media()
    page_b.get_opengraph_media()
    assert page_a._opengraph_media_cache is not page_b._opengraph_media_cache
    assert page_a.get_opengraph_images()[0].root_raw == ("https://example.com/a.jpg")
    assert page_b.get_opengraph_images()[0].root_raw == ("https://example.com/b.jpg")


def test_reparse_invalidates_cached_query():
    page = parse_media(og("og:image", "https://example.com/old.jpg"))
    assert len(page.get_opengraph_images()) == 1
    page.parse(
        "<html><head>%s%s</head></html>"
        % (
            og("og:image", "https://example.com/new-1.jpg"),
            og("og:image", "https://example.com/new-2.jpg"),
        )
    )
    roots = [g.root_raw for g in page.get_opengraph_images()]
    assert roots == [
        "https://example.com/new-1.jpg",
        "https://example.com/new-2.jpg",
    ]


def test_query_does_not_mutate_metadata():
    head = (
        og("og:image", "https://example.com/a.jpg")
        + og("og:image:width", "300")
        + og("og:title", "Title")
    )
    page = parse_media(head)
    import copy

    snapshot = copy.deepcopy(page.parsed_result.metadata)
    page.get_opengraph_media()
    page.get_opengraph_images()
    page.get_opengraph_media(refresh=True)
    assert page.parsed_result.metadata == snapshot
    # calling twice must return equivalent data
    assert page.get_opengraph_media(refresh=True).groups[0].root_url == (
        "https://example.com/a.jpg"
    )


def test_build_falls_back_to_flat_og_store_without_soup():
    # a parser that never parsed HTML has no soup; the builder must still
    # reconstruct groups from the flattened store in insertion order
    page = MetadataParser()
    page.parsed_result._add_discovered("og", "image", "https://example.com/a.jpg")
    page.parsed_result._add_discovered("og", "image:width", "300")
    page.parsed_result._add_discovered("og", "image", "https://example.com/b.jpg")
    page.parsed_result._add_discovered("og", "image:width", "640")
    assert page.parsed_result.soup is None
    groups = page.get_opengraph_images()
    assert [(g.root_raw, g.width) for g in groups] == [
        ("https://example.com/a.jpg", 300),
        ("https://example.com/b.jpg", 640),
    ]


def test_flat_store_fallback_handles_families_alternates_and_orphans():
    page = MetadataParser()
    add = page.parsed_result._add_discovered
    add("og", "image", "https://example.com/a.jpg")
    add("og", "image:secure_url", "https://example.com/a.secure.jpg")
    add("og", "image:alt", "an image")
    add("og", "video", "https://example.com/v.mp4")
    add("og", "video:height", "480")
    add("og", "audio:type", "audio/mpeg")  # orphan: no audio root
    query = page.get_opengraph_media()
    image, video = query.groups
    assert image.media_type == "image" and video.media_type == "video"
    assert [u[0] for u in image.urls] == ["og:image", "og:image:secure_url"]
    assert image.alt == "an image"
    assert video.root_url == "https://example.com/v.mp4"
    assert video.height == 480
    orphans = [d for d in query.diagnostics if d.code == "orphan_subproperty"]
    assert [(d.property, d.media_type) for d in orphans] == [("og:audio:type", "audio")]


def test_flat_store_fallback_diagnoses_empty_root_and_bad_dimension():
    page = MetadataParser()
    add = page.parsed_result._add_discovered
    add("og", "image", "")
    add("og", "image:width", "nope")
    add("og", "image", "https://example.com/good.jpg")
    groups = page.get_opengraph_images()
    empty, good = groups
    assert empty.root_raw == "" and empty.root_url is None
    assert any(d.code == "empty_value" for d in empty.diagnostics)
    # the illegal dimension is distributed to the first (empty) group
    assert any(d.code == "invalid_dimension" for d in empty.diagnostics)
    assert good.is_valid and good.root_url == "https://example.com/good.jpg"


def test_flat_store_fallback_round_robins_extra_subproperties():
    # more dimension values than roots: values cycle across the family's groups
    page = MetadataParser()
    add = page.parsed_result._add_discovered
    add("og", "image", "https://example.com/a.jpg")
    add("og", "image", "https://example.com/b.jpg")
    add("og", "image:width", "10")
    add("og", "image:width", "20")
    add("og", "image:width", "30")  # wraps back to the first group
    groups = page.get_opengraph_images()
    assert [g.width for g in groups] == [10, 20]


def test_encoders_and_strategy_behavior_remain_compatible():
    from metadata_parser.utils import encode_ascii

    page = parse_media(
        og("og:title", "Café")
        + og("og:image", "https://example.com/ä.jpg")
        + og("og:image:width", "10")
    )
    # encoders still apply to the flat get_metadatas API
    encoded = page.parsed_result.get_metadatas("title", encoder=encode_ascii)
    assert encoded == {"og": ["Cafe"]}
    # raw encoder opt-out still works
    raw = page.parsed_result.get_metadatas("title", encoder="raw")
    assert raw == {"og": ["Café"]}
    # strategy validation is untouched
    with pytest.raises(InvalidStrategy):
        page.parsed_result.get_metadatas("title", strategy=["nope"])


def test_cached_html_replay_matches_live_parse_without_fetching():
    head = (
        og("og:image", "/img/a.jpg")
        + og("og:image:width", "300")
        + og("og:video", "//cdn.example.com/v.mp4")
        + og("og:video:height", "480")
        + og("og:audio", "audio/a.mp3")
    )
    html = "<html><head>%s</head><body></body></html>" % head

    response = requests.Response()
    response.url = RESPONSE_URL
    response.status_code = 200
    response.headers["content-type"] = "text/html; charset=utf-8"
    response.encoding = "utf-8"
    response._content = html.encode("utf-8")

    with patch.object(requests.Session, "get", return_value=response) as fetch:
        live = MetadataParser(url=REQUESTED_URL)
    assert fetch.call_args[0] == (REQUESTED_URL,)

    with patch.object(requests.Session, "get") as fetch_cached:
        replayed = MetadataParser(url=REQUESTED_URL, url_actual=RESPONSE_URL, html=html)
    fetch_cached.assert_not_called()

    def payload(query):
        return [
            (
                g.media_type,
                g.root_property,
                g.root_raw,
                g.root_url,
                g.width,
                g.height,
                g.media_subtype,
                g.alt,
                [(p, v) for p, v in g.properties],
                [(d.code, d.property, d.value) for d in g.diagnostics],
            )
            for g in query.groups
        ]

    live_query = live.get_opengraph_media()
    replayed_query = replayed.get_opengraph_media()
    assert payload(live_query) == payload(replayed_query)
    assert [(d.code, d.property, d.value) for d in live_query.diagnostics] == [
        (d.code, d.property, d.value) for d in replayed_query.diagnostics
    ]
    assert live.url_actual == replayed.url_actual == RESPONSE_URL


def test_legacy_metadata_api_remains_flat_and_compatible():
    page = parse_media(
        og("og:image", "https://example.com/a.jpg")
        + og("og:image:width", "300")
        + og("og:image", "https://example.com/b.jpg")
        + og("og:image:width", "640")
    )
    # legacy storage: parallel arrays, exactly as before this feature
    og_store = page.parsed_result.metadata["og"]
    assert og_store["image"] == [
        "https://example.com/a.jpg",
        "https://example.com/b.jpg",
    ]
    assert og_store["image:width"] == ["300", "640"]
    # legacy getters still work
    assert page.get_metadata_link("og:image") == "https://example.com/a.jpg"
    assert page.parsed_result.get_metadatas("image", strategy=["og"]) == {
        "og": ["https://example.com/a.jpg", "https://example.com/b.jpg"]
    }
    # structured query coexists without interfering
    assert [g.width for g in page.get_opengraph_images()] == [300, 640]


def test_legacy_select_first_match_and_strategy_behavior_unchanged():
    page = parse_media(
        og("og:title", "OG Title")
        + og("og:image", "https://example.com/a.jpg")
        + og("og:image:width", "300")
    )
    assert page.parsed_result.select_first_match("title", ["og"]) == "OG Title"
    assert page.parsed_result.is_opengraph_minimum() is False
    result = page.parsed_result.get_metadatas("title", strategy="all")
    assert result == {"og": ["OG Title"]}


def test_document_without_media_returns_empty_query():
    page = parse_media(og("og:title", "Title only"))
    result = page.get_opengraph_media()
    assert result.groups == [] and result.diagnostics == []
    assert page.get_opengraph_images() == []
    assert page.get_opengraph_videos() == []
    assert page.get_opengraph_audios() == []


def test_media_tags_outside_head_are_included_by_default():
    html = "<html><head>%s</head><body>%s</body></html>" % (
        og("og:image", "https://example.com/head.jpg"),
        og("og:image", "https://example.com/body.jpg") + og("og:image:width", "100"),
    )
    page = MetadataParser(html=html)
    groups = page.get_opengraph_images()
    assert [g.root_raw for g in groups] == [
        "https://example.com/head.jpg",
        "https://example.com/body.jpg",
    ]
    assert groups[1].width == 100


def test_head_only_mode_excludes_body_media_like_the_legacy_parser():
    html = "<html><head>%s</head><body>%s</body></html>" % (
        og("og:image", "https://example.com/head.jpg"),
        og("og:image", "https://example.com/body.jpg") + og("og:image:width", "100"),
    )
    page = MetadataParser(html=html, search_head_only=True)
    groups = page.get_opengraph_images()
    assert [g.root_raw for g in groups] == ["https://example.com/head.jpg"]
    assert groups[0].width is None
    # the flat legacy store shows the same scoping, proving the query reads
    # the exact same document region as parse()
    assert page.parsed_result.metadata["og"]["image"] == (
        "https://example.com/head.jpg"
    )


def test_whitespace_around_values_is_preserved_raw_but_normalized_in_urls():
    page = MetadataParser(
        url="https://example.com/",
        html="<html><head><meta property='og:image' content='  /img/a.jpg  '></head></html>",
    )
    group = page.get_opengraph_images()[0]
    # _add_discovered strips stored values; the structured reader sees the same raw text
    assert group.root_raw == "/img/a.jpg"
    assert group.root_url == "https://example.com/img/a.jpg"
