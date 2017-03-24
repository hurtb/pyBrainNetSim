import matplotlib.pyplot as plt
import pyBrainNetSim.drawing.viewers as vis
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

plt.rcParams['animation.ffmpeg_path'] = 'C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe'
writer = animation.FFMpegWriter(fps=4, bitrate=1000)

def animate(world, individual_id, fname=None, *args, **kwargs):
    simnet = world.individuals[individual_id].internal
    fig = plt.gcf()
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(left=0.05, right=0.95, wspace=0.05)
    ax1 = plt.subplot(gs1[:-1, :-2])
    ax2 = plt.subplot(gs1[:-1, 2:])
    ax3 = plt.subplot(gs1[-1, :-3])
    ax4 = plt.subplot(gs1[-1, 1:])

    def animate_frame(i):
        _tmp = [ax.cla() for ax in [ax1, ax2, ax3, ax4]]
        fig.suptitle('Network %d Neurons (%d%% excitatory) @t%s'
                     % (simnet.simdata[i].number_of_nodes(), 100.* simnet.simdata[i].excitatory_to_inhibitory_ratio , i), fontsize=20)
        vis.draw_networkx(simnet.simdata[i], ax=ax1)
        world.plot_individual_trajectory(individual=individual_id,
                                         at_time=i,
                                         upsample_factor=10,
                                         ax=ax2)
        vis.pcolormesh_edge_changes(simnet.simdata,
                                    initial_time=0,
                                    final_time=i,
                                    as_pct=True,
                                    ax=ax3)
        vis.degree_histogram(simnet.simdata[i], min_weight=.5, ax=ax4)
        ax1.set_title('Network @t=%d' % i)
        ax2.set_title('Environment @t=%d' % i)
        ax1.axis('off')
        ax2.axis('off')
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])

    ani = animation.FuncAnimation(fig, animate_frame, len(simnet.simdata), repeat=False, interval=20)
    if isinstance(fname, str):
        ani.save('%s.mp4' %fname, writer=writer)

    # plt.show()