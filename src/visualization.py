import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
import polars as pl
import numpy as np

def create_football_field(linenumbers=True,
                          endzones=True,
                          highlight_line=False,
                          highlight_line_number=50,
                          highlighted_name='Line of Scrimmage',
                          fifty_is_los=False,
                          figsize=(12, 6.33)):
    """
    Creates a rectangle with the football field.
    """
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                             edgecolor='r', facecolor='darkgreen', zorder=0)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 0],
             color='white')
    
    if fifty_is_los:
        plt.plot([60, 60], [0, 53.3], color='gold')
        plt.text(62, 50, '<- Player Direction', color='gold')

    if endzones:
        ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ax.add_patch(ez1)
        ax.add_patch(ez2)

    plt.xlim(0, 120)
    plt.ylim(-5, 58.3)
    plt.axis('off')

    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            plt.text(x, 5, str(numb - 10),
                     horizontalalignment='center',
                     verticalalignment='center',
                     color='white')
            plt.text(x - 0.95, 53.3 - 5, str(numb - 10),
                     horizontalalignment='center',
                     verticalalignment='center',
                     color='white', rotation=180)

    if highlight_line:
        hl = highlight_line_number + 10
        plt.plot([hl, hl], [0, 53.3], color='yellow')
        plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),
                 color='yellow')
                 
    return fig, ax

def plot_play_frame(play_df, frame_id, ax=None):
    """
    Plots players for a specific frame.
    """
    if ax is None:
        fig, ax = create_football_field()
        
    frame_data = play_df.filter(pl.col("frame_id") == frame_id)
    
    # Assume 'club' distinguishes teams.
    # We need to map club to colors.
    clubs = frame_data["club"].unique().to_list()
    # Simple color map
    colors = {clubs[0]: 'red', clubs[1]: 'blue'} if len(clubs) > 1 else {clubs[0]: 'red'}
    if 'football' in clubs: # Football usually has club 'football' or nan?
         # Check data loader output for standard football club name
         pass
         
    # Need to convert to pandas for plotting loop
    pdf = frame_data.to_pandas()
    
    for _, row in pdf.iterrows():
        color = 'white'
        size = 50
        marker = 'o'
        
        if row['player_name'] == 'football':
            color = 'brown'
            marker = 'd' # diamond
            size = 30
        elif row['club'] in colors:
            color = colors[row['club']]
        else:
            color = 'grey' # Default
            
        ax.scatter(row['std_x'], row['std_y'], s=size, c=color, marker=marker, edgecolors='black', zorder=10)
        
        # Plot orientation if available
        # o is degrees, 0 is usually facing X axis?
        # Actually in NFL data: 0 is facing visitor sideline (y=53.3) or similar.
        # Need to check standardization.
        # Let's verify 'dir' vector usually better for movement.
        
    return ax

def animate_play(play_df, output_path="play_animation.mp4"):
    """
    Creates an animation of the play.
    """
    frames = play_df["frame_id"].unique().sort().to_list()
    
    fig, ax = create_football_field()
    
    def update(frame):
        ax.clear()
        create_football_field(figsize=(12, 6.33)) # Background logic needs refactor to avoid redrawing
        # Better: create background once, update scatter
        # For simplicity of this sprint, just redraw elements or use fixed background
        
        # Just re-call create_football_field logic on same ax? 
        # Actually create_football_field returns new fig/ax.
        # Let's fix this structure for animation.
        pass 
        
    # Re-writing for proper animation structure
    plt.close(fig)
    fig, ax = create_football_field()
    
    scat = ax.scatter([], [], s=50, zorder=10, edgecolors='black')
    
    # Pre-process data
    pdf = play_df.to_pandas()
    
    unique_clubs = [c for c in pdf['club'].unique() if c != 'football']
    colors = {c: 'red' if i == 0 else 'blue' for i, c in enumerate(unique_clubs)}
    colors['football'] = 'brown'
    
    def init():
        return (scat,)
        
    def animate(frame_val):
        frame_data = pdf[pdf['frame_id'] == frame_val]
        
        offsets = []
        facecolors = []
        
        for _, row in frame_data.iterrows():
            offsets.append([row['std_x'], row['std_y']])
            # Color
            c_key = row['club'] if row['player_name'] != 'football' else 'football'
            # club might be nan for football
            if row['player_name'] == 'football':
                 c_key = 'football'
            
            facecolors.append(colors.get(c_key, 'white'))
            
        scat.set_offsets(offsets)
        scat.set_facecolors(facecolors)
        ax.set_title(f"Frame: {frame_val}")
        return (scat,)
        
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=frames, interval=100, blit=True)
                                   
    if output_path.endswith('.gif'):
        anim.save(output_path, writer='pillow', fps=10)
    else:
        # Default to ffmpeg for mp4
        anim.save(output_path, writer='ffmpeg', fps=10)
        
    plt.close()
    return output_path
