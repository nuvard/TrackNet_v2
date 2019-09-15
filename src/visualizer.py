import matplotlib

if not 'inline' in matplotlib.get_backend():
    matplotlib.use('Qt5Agg')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.cm as cm
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import mpl_toolkits.mplot3d.art3d as art3d

from collections import namedtuple

def namedtuple_me(s, name='PNT'):
    return namedtuple(name, s.index)(*s)

def revert_types(df):
    df = df.astype({'track':'int32', 'station':'int32', 'event':'int32'})
    df.index = df.index.map(int)
    return df

class Visualizer:

    def __init__(self, df = None, title = "EVENT GRAPH"):
        self.__df = df
        self.__axs = []
        self.__color_map = {-1: [[0.1, 0.1, 0.1]]}
        self.__adj_track_list = []
        self.__reco_adj_list = []
        self.__fake_hits = []
        self.__nn_preds = []
        self.__coord_planes = []
        self.__draw_all_hits = False
        self.__draw_all_tracks_from_df = False
        self.__title = title

    def init_draw(self, reco_tracks = None, draw_all_tracks_from_df = False, draw_all_hits = False):
        self.__draw_all_hits = draw_all_hits
        self.__draw_all_tracks_from_df = draw_all_tracks_from_df
        grouped = self.__df.groupby('track')
        # prepare adjacency list for tracks
        for i, gp in grouped:
            if gp.track.values[0] == -1:
                self.__fake_hits.extend(gp[['station','x','y']].values)
                continue
            for row in range(1, len(gp.index)):
                elem = (gp.index[row - 1], gp.index[row], 1)
                self.__adj_track_list.append(elem)

        if reco_tracks is not None:
            for track in reco_tracks:
                for i in range(0, len(track) - 2):
                    if track[i] == -1 or track[i+1] == -1:
                        break
                    self.__reco_adj_list.append((track[i], track[i + 1], 1))

    def add_nn_pred(self, z_ell_coord, from_idx, pred_X_Y_Station, pred_R1_R2):
         self.__nn_preds.append([z_ell_coord,from_idx, pred_X_Y_Station, pred_R1_R2])

    def add_coord_planes(self, coord_planes_arr):
        self.__coord_planes = coord_planes_arr

    def set_title(self, title = "EVENT GRAPH"):
        self.__title = title

    def draw(self, show=True):
        matplotlib.rcParams['legend.fontsize'] = 10
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(self.__title)
        ax.set_xlabel('Station')
        ax.set_ylabel('X')
        ax.set_zlabel('Y')
        legends = {}
        if self.__draw_all_hits:
            for fake_hit in self.__fake_hits:
                ax.scatter(fake_hit[0], fake_hit[1], fake_hit[2], c=self.__color_map[-1], marker='o')

        if self.__draw_all_tracks_from_df:
            for adj_val in self.__adj_track_list:
                col, lab, tr_id =self.draw_edge(adj_val, ax)
                if int(tr_id) not in legends:
                    legends[int(tr_id)] = mpatches.Patch(color=col, label=lab)

        for adj_val in self.__reco_adj_list:
            col, lab, tr_id = self.draw_edge(adj_val, ax)
            if int(tr_id) not in legends:
                legends[int(tr_id)] = mpatches.Patch(color=col, label=lab)


        for ell_data in self.__nn_preds:
            ell = Ellipse(xy=ell_data[2], width=ell_data[3][0], height=ell_data[3][1], color='red')
            ax.add_patch(ell)
            art3d.pathpatch_2d_to_3d(ell, z=ell_data[0], zdir="x")
            col, lab, tr_id = self.draw_edge_from_idx_to_pnt(ell_data[1], [ell_data[0], ell_data[2][0], ell_data[2][1]], ax)
            if int(tr_id) not in legends:
                legends[int(tr_id)] = mpatches.Patch(color=col, label=lab)

        for station_id, coord_planes in enumerate(self.__coord_planes):
            for rect_data in coord_planes:
                rect = Rectangle(xy=(rect_data[0]-rect_data[2]/2, rect_data[1]-rect_data[3]/2) , width=rect_data[2], height=rect_data[3], linewidth=1, edgecolor='black',facecolor='none')
                ax.add_patch(rect)
                art3d.pathpatch_2d_to_3d(rect, z=station_id, zdir="x")

        fig.legend(handles=list(legends.values()))
        # for single_data, with_edges, with_pnts, title in self.__all_data:
        #     legends = {}
        #     for (fr_p, to_p, dist) in single_data:
        #         color, label, tr_id = self.generate_color_label(fr_p.track, to_p.track)
        #         if with_edges:
        #             ax.plot((fr_p.station, to_p.station), (fr_p.x, to_p.x), zs=(fr_p.y, to_p.y),
        #                     c=color)
        #         marker_1 = 'h' if fr_p.track == -1 else 'o'
        #         marker_2 = 'h' if to_p.track == -1 else 'o'
        #         if with_pnts:
        #             ax.scatter(fr_p.station, fr_p.x, fr_p.y, c =self.__color_map[int(fr_p.track)], marker=marker_1)
        #             ax.scatter(to_p.station, to_p.x, to_p.y, c =self.__color_map[int(to_p.track)], marker=marker_2)
        #         if int(tr_id) not in legends:
        #             legends[int(tr_id)] = mpatches.Patch(color=color, label=label)
        #     fig.legend(handles=list(legends.values()))

        plt.draw_all()
        plt.tight_layout()
        if show:
            plt.show()
        pass

    def draw_edge(self, adj_val, ax):
        hit_from = self.__df.loc[adj_val[0]]
        hit_to = self.__df.loc[adj_val[1]]
        color, label, tr_id = self.generate_color_label(int(hit_from.track), int(hit_to.track))
        marker_1 = 'h' if hit_from.track == -1 else 'o'
        marker_2 = 'h' if hit_to.track == -1 else 'o'
        ax.plot((hit_from.station, hit_to.station), (hit_from.x, hit_to.x), zs=(hit_from.y, hit_to.y), c=color)
        ax.scatter(hit_from.station, hit_from.x, hit_from.y, c=self.__color_map[int(hit_from.track)], marker=marker_1)
        ax.scatter(hit_to.station, hit_to.x, hit_to.y, c=self.__color_map[int(hit_to.track)], marker=marker_2)
        return color, label, tr_id

    def draw_edge_from_idx_to_pnt(self, from_idx,
                                  to_coord_STATXY, ax,
                                  line_color=np.random.rand(3,),
                                  marker='h',
                                  pnt_color='yellow'):
        hit_from = self.__df.loc[from_idx]
        ax.plot((hit_from.station, to_coord_STATXY[0]), (hit_from.x, to_coord_STATXY[1]),
                zs=(hit_from.y, to_coord_STATXY[2]), c=line_color)
        ax.scatter(to_coord_STATXY[0], to_coord_STATXY[1], to_coord_STATXY[2],
                   c=pnt_color, marker=marker)
        return line_color, 'test edge from ev_id:' + str(int(hit_from.track)), int(hit_from.track)

    def redraw_all(self):
        pass

    def generate_color_label(self, tr_id_from, tr_id_to):
        if tr_id_from not in self.__color_map:
            self.__color_map[tr_id_from] = np.random.rand(3,)
        if tr_id_to not in self.__color_map:
            self.__color_map[tr_id_to] = np.random.rand(3,)
        if tr_id_from != tr_id_to:
            return (1, 0.1, 0.1), 'bad connection', tr_id_from<<16|tr_id_to
        if tr_id_from == -1:
            return (0.1, 0.1, 0.1), 'fake connection', -1
        return self.__color_map[tr_id_from], 'tr_id: ' + str(int(tr_id_from)), tr_id_from