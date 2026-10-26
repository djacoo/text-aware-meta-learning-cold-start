import json

import pytest

from coldstart.download import FILES
from coldstart.preprocess import load_movielens

MOVIES = [
    "1::Movie A::1995::PG::x::Drama, Comedy::Ann Lee::x::Bo Ray, Cy Dee::A plot.::x",
    "2::Movie B::1996::R::x::Drama::Ann Lee, Dan Moe(co-director)::x::Bo Ray::B plot.::x",
    "3::Movie C::1997::PG::x::N/A::N/A::x::N/A::N/A::x",
]
VOCABS = {
    "m_rate.txt": ["PG", "R"],
    "m_genre.txt": ["Comedy", "Drama"],
    "m_director.txt": ["Ann Lee", "Dan Moe"],
    "m_actor.txt": ["Bo Ray", "Cy Dee"],
    "m_gender.txt": ["F", "M"],
    "m_age.txt": ["1", "18"],
    "m_occupation.txt": ["0", "1"],
    "m_zipcode.txt": ["12345"],
}


@pytest.fixture
def ml_root(tmp_path):
    (tmp_path / "movies_extrainfos.dat").write_text("\n".join(MOVIES))
    users = [f"{u}::{'F' if u % 2 else 'M'}::18::1::12345-678" for u in range(1, 21)]
    (tmp_path / "users.dat").write_text("\n".join(users))
    for name, values in VOCABS.items():
        (tmp_path / name).write_text("\n".join(values))

    # user u has 10 + u ratings, so users 1 and 2 fall under the 13 ratings minimum
    movies = {str(u): [str(1 + i % 3) for i in range(10 + u)] for u in range(1, 21)}
    ratings = {u: [1 + i % 5 for i in range(len(m))] for u, m in movies.items()}
    for state in ("warm_state", "user_cold_state", "item_cold_state", "user_and_item_cold_state"):
        (tmp_path / f"{state}.json").write_text(json.dumps(movies))
        (tmp_path / f"{state}_y.json").write_text(json.dumps(ratings))

    assert all((tmp_path / name).exists() for name in FILES)
    return tmp_path


@pytest.fixture
def data(ml_root):
    return load_movielens(ml_root)
