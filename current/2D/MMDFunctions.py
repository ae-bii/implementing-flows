# Copyright 2021 Thomas Viehmann

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modification in line 50
# Modification Copyright 2022 Alfin Hou




import torch
import tqdm

def mmd(x, y, sigma):


    n, d = x.shape
    m, d2 = y.shape
    assert d == d2
    xy = torch.cat([x.detach(), y.detach()], dim=0)
    dists = torch.cdist(xy, xy, p=2.0)

    k = torch.exp((-1/(2*sigma**2)) * dists**2) + torch.eye(n+m)*1e-5
    k_x = k[:n, :n]
    k_y = k[n:, n:]
    k_xy = k[:n, n:]

    mmd = k_x.sum() / (n * (n - 1)) + k_y.sum() / (m * (m - 1)) - 2 * k_xy.sum() / (n * m)
    return mmd

def TestThreshold(x,y,sigma):

    N_X = len(x)
    N_Y = len(y)
    xy = torch.cat([x, y], dim=0)[:, None].double()

    mmds = []
    for i in tqdm.tqdm(range(1000)):
        xy = xy[torch.randperm(len(xy))]
        xy = xy.squeeze(1)
        mmds.append(mmd(xy[:N_X], xy[N_X:], sigma).item())
    mmds = torch.tensor(mmds)
    return torch.quantile(mmds,0.95)

