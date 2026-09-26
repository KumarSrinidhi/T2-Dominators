"""Track stitching (argonix/stitch.py): a briefly lost person keeps one id; different people do not merge."""
from argonix.detect import OBJECT_ID_OFFSET
from argonix.stitch import StitchedBackend, TrackStitcher, wrap_factory


def person(tid, x, y=300, h=100):
    return (tid, (x - 20, y - h, x + 20, y), "person", 0.9)


def test_occlusion_gap_keeps_the_same_id():
    s = TrackStitcher()
    ids = [s.update([person(1, 100 + 10 * f)])[0][0] for f in range(5)]   # walking right, 10 px per frame
    for _ in range(3):                                                     # occluded for 3 frames
        s.update([])
    back = s.update([person(7, 180)])                                      # tracker gives a NEW raw id
    assert back[0][0] == ids[0] and len(set(ids)) == 1 and s.stats["stitched"] == 1


def test_far_away_or_different_size_is_a_new_person():
    s = TrackStitcher()
    first = s.update([person(1, 100)])[0][0]
    s.update([])
    assert s.update([person(2, 600)])[0][0] != first                       # appears far away
    s2 = TrackStitcher()
    a = s2.update([person(1, 100, h=100)])[0][0]
    s2.update([])
    assert s2.update([person(2, 105, h=30)])[0][0] != a                    # a much smaller (farther) person


def test_two_visible_people_never_share_an_id_and_old_gaps_expire():
    s = TrackStitcher(max_gap_frames=3)
    out = s.update([person(1, 100), person(2, 110)])
    assert out[0][0] != out[1][0]
    lost = s.update([person(2, 110)])[0][0]
    for _ in range(5):                                                     # longer than the gap: forgotten
        s.update([person(2, 110)])
    again = s.update([person(2, 110), person(9, 100)])
    assert again[0][0] == lost and again[1][0] not in (lost, out[0][0])


def test_objects_pass_through_and_the_factory_wraps_only_when_enabled():
    s = TrackStitcher()
    obj = (OBJECT_ID_OFFSET + 5, (0, 0, 10, 10), "backpack", 0.5)
    assert s.update([obj]) == [obj]

    class B:
        info = {"detector": "fake"}

        def track(self, frame):
            return [person(3, 50)]

        def close(self):
            return "closed"
    fac = lambda cam: B()  # noqa: E731
    assert wrap_factory(fac, {"stitch": {"enabled": False}}) is fac          # opt-out
    assert isinstance(wrap_factory(fac, {})({"id": "c"}), StitchedBackend)   # on by default
    w = wrap_factory(fac, {"stitch": {"max_gap_frames": 10}})({"id": "c"})
    assert isinstance(w, StitchedBackend) and w.track(None)[0][0] == 1 and w.close() == "closed"
    assert w.info["stitch"]["max_gap_frames"] == 10
