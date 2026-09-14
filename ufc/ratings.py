"""Rating systems updated strictly chronologically. All return PRE-fight values before `update`."""
import math


def elo_expect(ra, rb, scale=400.0):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / scale))


class Elo:
    """Win/loss Elo. `finish_mult` scales K for finishes (margin of victory); `debut` lets callers seed ratings."""

    def __init__(self, k=40.0, init=1500.0, finish_mult=1.0, k_decay_fights=0.0):
        self.k, self.init, self.finish_mult, self.kdf = k, init, finish_mult, k_decay_fights
        self.r, self.n = {}, {}

    def get(self, f):
        return self.r.get(f, self.init)

    def update(self, a, b, score_a, finish=False):
        ra, rb = self.get(a), self.get(b)
        e = elo_expect(ra, rb)
        mult = self.finish_mult if finish else 1.0
        ka = self.k * mult * (1 + self.kdf / (1 + self.n.get(a, 0)))  # bigger steps while a fighter is new
        kb = self.k * mult * (1 + self.kdf / (1 + self.n.get(b, 0)))
        self.r[a] = ra + ka * (score_a - e)
        self.r[b] = rb - kb * (score_a - e)
        self.n[a] = self.n.get(a, 0) + 1
        self.n[b] = self.n.get(b, 0) + 1


class Glicko2:
    """Glicko-2 with rating deviation growing during inactivity (per 30-day period)."""

    def __init__(self, tau=0.5, init_rd=350.0, init_vol=0.06, period_days=30.0):
        self.tau, self.init_rd, self.init_vol, self.period = tau, init_rd, init_vol, period_days
        self.s = {}  # fighter -> (mu, phi, sigma, last_date)

    def _cur(self, f, date):
        if f not in self.s:
            return 0.0, self.init_rd / 173.7178, self.init_vol
        mu, phi, sig, last = self.s[f]
        t = max((date - last).days / self.period, 0.0)
        phi = min(math.sqrt(phi * phi + t * sig * sig), self.init_rd / 173.7178)
        return mu, phi, sig

    def get(self, f, date):
        mu, phi, _ = self._cur(f, date)
        return 1500 + 173.7178 * mu, 173.7178 * phi

    @staticmethod
    def _g(phi):
        return 1.0 / math.sqrt(1 + 3 * phi * phi / math.pi ** 2)

    def expect(self, a, b, date):
        mua, phia, _ = self._cur(a, date)
        mub, phib, _ = self._cur(b, date)
        return 1.0 / (1.0 + math.exp(-self._g(math.sqrt(phia ** 2 + phib ** 2)) * (mua - mub)))

    def _one(self, mu, phi, sig, muj, phij, s):
        g = self._g(phij)
        e = 1.0 / (1.0 + math.exp(-g * (mu - muj)))
        v = 1.0 / (g * g * e * (1 - e))
        delta = v * g * (s - e)
        a = math.log(sig * sig)
        tau = self.tau
        f = lambda x: math.exp(x) * (delta ** 2 - phi ** 2 - v - math.exp(x)) / (2 * (phi ** 2 + v + math.exp(x)) ** 2) - (x - a) / tau ** 2
        A = a
        if delta ** 2 > phi ** 2 + v:
            B = math.log(delta ** 2 - phi ** 2 - v)
        else:
            k = 1
            while f(a - k * tau) < 0:
                k += 1
            B = a - k * tau
        fa, fb = f(A), f(B)
        for _ in range(60):
            if abs(B - A) < 1e-6:
                break
            C = A + (A - B) * fa / (fb - fa)
            fc = f(C)
            if fc * fb <= 0:
                A, fa = B, fb
            else:
                fa /= 2
            B, fb = C, fc
        sig2 = math.exp(A / 2)
        phistar = math.sqrt(phi ** 2 + sig2 ** 2)
        phi2 = 1 / math.sqrt(1 / phistar ** 2 + 1 / v)
        mu2 = mu + phi2 ** 2 * g * (s - e)
        return mu2, phi2, sig2

    def update(self, a, b, score_a, date):
        mua, phia, siga = self._cur(a, date)
        mub, phib, sigb = self._cur(b, date)
        self.s[a] = (*self._one(mua, phia, siga, mub, phib, score_a), date)
        self.s[b] = (*self._one(mub, phib, sigb, mua, phia, 1 - score_a), date)


class ShareElo:
    """Elo on a continuous share in [0,1] (e.g. significant-strike share). Opponent-adjusted skill rating."""

    def __init__(self, k=60.0, init=1500.0):
        self.k, self.init, self.r = k, init, {}

    def get(self, f):
        return self.r.get(f, self.init)

    def update(self, a, b, share_a):
        if share_a is None or share_a != share_a:
            return
        ra, rb = self.get(a), self.get(b)
        d = self.k * (share_a - elo_expect(ra, rb))
        self.r[a], self.r[b] = ra + d, rb - d


if __name__ == "__main__":
    import datetime as dt
    e = Elo(k=32); e.update("a", "b", 1)
    assert e.get("a") == 1516 and e.get("b") == 1484
    g = Glicko2(); d = dt.date(2020, 1, 1)
    g.update("a", "b", 1, d)
    ra, rda = g.get("a", d); rb, _ = g.get("b", d)
    assert ra > 1500 > rb and rda < 350
    assert g.get("a", dt.date(2025, 1, 1))[1] > rda  # inactivity widens uncertainty
    print("ratings ok")
