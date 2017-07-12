from flask import Flask, request, render_template
import flask.templating

import json
from functools import partial

import requests
import numpy as np
import pandas as pd
import pathos.multiprocessing as mp

import mpld3
import matplotlib.pyplot as plt
import matplotlib as mpl

import bokeh
from bokeh.models.sources import ColumnDataSource
from bokeh.plotting import figure
from bokeh.resources import INLINE
from bokeh.embed import components
from bokeh.util.string import encode_utf8

from dict_utils import flatten

import renderapi

app = Flask(__name__, static_folder='static')


@app.route('/')
def hello_world():
    return 'Hello, World!'


def process_group(render, matchcollections, group):
    matches = []

    matches = renderapi.pointmatch.get_matches_outside_group(
        matchcollections[0], group,
        mergeCollections=matchcollections[1:], render=render)

    matchcounts = {}
    for match in matches:
        qid = match['qGroupId']
        pid = match['pGroupId']
        if int(qid) == group:
            otherid = int(pid)
        else:
            otherid = int(qid)
        # j=np.where(groups==otherid)[0][0]
        nummatches = len(match['matches']['p'][0])
        nummatches = np.clip(nummatches, 0, 10)
        if otherid in matchcounts.keys():
            matchcounts[otherid] += nummatches
        else:
            matchcounts[otherid] = nummatches
    return matchcounts


def get_groups(render, matchcollection):
    groups = renderapi.pointmatch.get_match_groupIds(
        matchcollection, render=render)
    groups = np.array(map(int, groups))
    groups.sort()
    return groups


def get_merged_groups(render, matchcollections):
    groups = np.array([])

    for mc in matchcollections:
        groups = np.append(get_groups(render, mc), groups)
    groups = list(set(groups))
    groups.sort()
    groups = np.array(groups, np.int)
    return groups


def get_z_value_dict(render, stack, answers, groups):
    zvalues = {}
    for answer, group in zip(answers, groups):
        z = renderapi.stack.get_section_z_value(stack, group, render=render)
        zvalues[group] = int(z)
        for group2 in answer.keys():
            if group2 not in zvalues.keys():
                z = renderapi.stack.get_section_z_value(
                    stack, group2, render=render)
                zvalues[group2] = int(z)
    return zvalues


def assemble_match_matrix(answers, groups, zvalues, maxz, maxdz=12):

    match_matrix = np.zeros((maxz + 1, 2 * maxdz + 1))

    for answer, group in zip(answers, groups):
        # print group,zvalues[group]
        z1 = zvalues[group]
        for group2 in answer.keys():
            z2 = zvalues[group2]
            val = answer[group2]
            if np.abs(z2 - z1) <= maxdz:
                match_matrix[z1, z2 - z1 + maxdz] += val
                match_matrix[z2, z1 - z2 + maxdz] += val
    return match_matrix


def make_pointmatch_summary_plot(render, stack, matchcollection,
                                 sections_per_row=200):
    matchcollections = [matchcollection]
    groups = get_merged_groups(render, matchcollections)
    with mp.ProcessingPool(5) as pool:
        mypartial = partial(process_group, render, matchcollections)
        answers = pool.map(mypartial, groups)

    zvalues = get_z_value_dict(render, stack, answers, groups)
    maxz = np.max(zvalues.values())
    match_matrix = assemble_match_matrix(answers, groups, zvalues)

    rows = int(np.ceil(maxz * 1.0 / sections_per_row))
    sections_per_row = int(np.ceil(maxz * 1.0 / rows))
    rows = int(np.ceil(maxz * 1.0 / sections_per_row))

    f, ax = plt.subplots(rows, 1, figsize=(8, 2 * rows))
    first_section_indices = np.concatenate(
        [np.array([0]), np.where(np.diff(groups % 1000) < 0)[0] + 1])
    first_sections = groups[first_section_indices]
    first_section_zs = [zvalues[section] for section in first_sections]

    maxval = np.max(np.log(match_matrix))

    for row in range(rows):
        startz = row * sections_per_row
        endz = row * sections_per_row + sections_per_row
        endz = min(endz, maxz)
        if rows > 1:
            theax = ax[row]
        else:
            theax = ax

        img = theax.imshow(np.log(match_matrix[startz:endz, :].T),
                           interpolation='nearest',
                           cmap=plt.cm.viridis,
                           extent=(startz - .5, endz - .5, maxdz, -maxdz), vmax=maxval)

        theax.autoscale(tight=True)
        for z in first_section_zs:
            if (z >= startz) & (z <= endz):
                theax.plot([z - .5, z - .5], [-maxdz, maxdz], c='w',
                           linewidth=2, linestyle='--', alpha=.5)
    return f


def make_pointmatch_plot(r, stack, matchcollection, z_p, z_q, max_d=40):
    section_p = renderapi.stack.get_sectionId_for_z(stack, z_p, render=r)
    section_q = renderapi.stack.get_sectionId_for_z(stack, z_q, render=r)
    allmatches = renderapi.pointmatch.get_matches_from_group_to_group(
        matchcollection, section_p, section_q, render=r)
    tilespecs_p = renderapi.tilespec.get_tile_specs_from_z(
        stack, z_p, render=r)
    tilespecs_q = renderapi.tilespec.get_tile_specs_from_z(
        stack, z_q, render=r)
    bounds_p = renderapi.stack.get_bounds_from_z(stack, z_p, render=r)

    all_points_global_p = np.zeros((1, 2))
    all_points_global_q = np.zeros((1, 2))
    for matchobj in allmatches:
        points_local_p = np.array(matchobj['matches']['p'])
        points_local_q = np.array(matchobj['matches']['q'])
        try:
            ts_p = next(ts for ts in tilespecs_p if ts.tileId ==
                        matchobj['pId'])
            ts_q = next(ts for ts in tilespecs_q if ts.tileId ==
                        matchobj['qId'])
        except:
            try:
                ts_p = next(ts for ts in tilespecs_p if ts.tileId ==
                            matchobj['qId'])
                ts_q = next(ts for ts in tilespecs_q if ts.tileId ==
                            matchobj['pId'])

            except:
                ts_p = None
                ts_q = None

        if ts_p is not None:

            t_p = ts_p.tforms[0].tform(points_local_p.T)
            t_q = ts_q.tforms[0].tform(points_local_q.T)
            all_points_global_p = np.append(all_points_global_p, t_p, axis=0)
            all_points_global_q = np.append(all_points_global_q, t_q, axis=0)

    all_points_global_p = all_points_global_p[1:, :]
    all_points_global_q = all_points_global_q[1:, :]
    all_points = np.concatenate(
        [all_points_global_p, all_points_global_q], axis=1)
    dv = np.sqrt(
        np.sum(np.power(all_points[:, 0:2] - all_points[:, 2:4], 2), axis=1))
    # f,ax=plt.subplots(1,1)
    # ax.scatter(all_points[:,0],all_points[:,1],c='m',marker='o',s=5,linewidth=0)
    # ax.quiver(all_points[:,0].T,all_points[:,1].T,
    #           all_points[:,2].T-all_points[:,0].T,
    #           all_points[:,3].T-all_points[:,1].T,
    #           color='m',
    #           angles='xy', scale_units='xy', scale=1)
    # ax.set_xlim((bounds_p['minX'],bounds_p['maxX']))
    # ax.set_ylim((bounds_p['maxY'],bounds_p['minY']))
    # ax.set_aspect('equal')
    # ax.set_title('%d_to_%d'%(z_p,z_q))
    # plt.tight_layout()
    # output = json.dumps(mpld3.fig_to_dict(f))
    # plt.close(f)
    xy = ColumnDataSource({'x0': all_points_global_p[:, 0],
                           'y0': all_points_global_p[:, 1],
                           'x1': all_points_global_q[:, 0],
                           'y1': all_points_global_q[:, 1],
                           'dv': dv})
    colors = ["#%02x%02x%02x" % (int(r_), int(g_), int(b_)) for r_, g_, b_, _
              in 255 * mpl.cm.cool(mpl.colors.Normalize(
                  vmin=0, vmax=max_d)(dv))]
    width_u = bounds_p['maxX'] - bounds_p['minX']
    height_u = bounds_p['maxY'] - bounds_p['minY']
    aspect_ratio = width_u * 1.0 / height_u
    if aspect_ratio > 1:
        width = 1000
        height = int(width / aspect_ratio)
    else:
        height = 1000
        width = int(height * aspect_ratio)
    print width, height
    plot = figure(plot_width=width,
                  plot_height=height,
                  y_range=(bounds_p['maxY'], bounds_p['minY']))
    plot.scatter('x0', 'y0', source=xy, color=colors)
    plot.segment('x0', 'y0', 'x1', 'y1', source=xy,
                 line_width=1, line_color=colors)

    return plot


def get_host_port(rendersource):

    parts = rendersource.split(':')
    host = parts[0]
    host = 'http://' + host
    port = int(parts[1].split('/')[0])
    return host, port


def package_bokeh_figure(fig):
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    script, div = components(fig)
    return script, div, js_resources, css_resources


def make_simple_matplotlib():
    f, ax = plt.subplots()
    ax.plot(np.random.random(10))
    return f


@app.route('/rendersource/<path:rendersource>/owner/<owner>/project/<project>'
           '/stack/<stack>/collection/<collection>/pointmatch')
def pointmatch_summary_view(rendersource, owner, project, stack, collection):
    host, port = get_host_port(rendersource)
    render = renderapi.render.connect(
        host, port, owner, project, client_scripts="")
    collections = renderapi.pointmatch.get_matchcollections(render=render)
    collectionData = [c for c in collections
                      if c['collectionId']['name'] == collection]
    numz = len(renderapi.stack.get_z_values_for_stack(stack, render=render))
    f = make_pointmatch_summary_plot(render, stack, collection)

    plot_json = json.dumps(mpld3.fig_to_dict(f))

    html = render_template(
        'embed_matplotlib.html',
        plot_json=plot_json
    )

    return html


@app.route('/rendersource/<path:rendersource>/owner/<owner>/project/<project>'
           '/stack/<stack>/collection/<collection>')
def collection_view(rendersource, owner, project, stack, collection):
    host, port = get_host_port(rendersource)
    render = renderapi.render.connect(
        host, port, owner, project, client_scripts="")
    collections = renderapi.pointmatch.get_matchcollections(render=render)
    collectionData = [c for c in collections
                      if c['collectionId']['name'] == collection]
    numz = len(renderapi.stack.get_z_values_for_stack(stack, render=render))
    # f=make_pointmatch_summary_plot(render,stack,matchcollection)

    f = make_simple_matplotlib()

    plot_json = json.dumps(mpld3.fig_to_dict(f))

    html = render_template(
        'embed_matplotlib.html',
        plot_json=plot_json
    )

    return html


@app.route('/rendersource/<path:rendersource>/owner/<owner>/project/<project>'
           '/stack/<stack>')
def stack_view(rendersource, owner, project, stack):
    host, port = get_host_port(rendersource)
    render = renderapi.render.connect(
        host, port, owner, project, client_scripts="")
    numz = len(renderapi.stack.get_z_values_for_stack(stack, render=render))
    collections = renderapi.pointmatch.get_matchcollections(render=render)

    return render_template('stack.html',
                           rendersource=rendersource,
                           owner=owner,
                           project=project,
                           stack=stack,
                           numz=numz,
                           collections=collections)


@app.route('/rendersource/<path:rendersource>/owner/<owner>/project/<project>')
def project_view(rendersource, owner, project):
    host, port = get_host_port(rendersource)
    render = renderapi.render.connect(
        host, port, owner, project, client_scripts="")
    # stacks = renderapi.render.get_stacks_by_owner_project(render=render)

    stackInfo = [s for s in renderapi.render.get_stack_metadata_by_owner(
        render=render) if s['stackId']['project'] == project]
    df = stackInfo_to_df(stackInfo, rendersource)

    return render_template('project.html',
                           rendersource=rendersource,
                           owner=owner,
                           project=project,
                           tables=[df.to_html(escape=False)],
                           titles=['stacks'])


def stackInfo_to_df(stackInfo, rendersource):
    pd.set_option('display.max_colwidth', -1)
    df = pd.io.json.json_normalize(stackInfo)
    df = df.sort_values('lastModifiedTimestamp', ascending=False)
    df.columns = df.columns.map(lambda x: x.split(".")[-1])
    df['stack'] = df.apply(
        lambda x: ("<a href='/rendersource/%s/owner/%s/"
                   "project/%s/stack/%s'> %s </a>" %
                   (rendersource, x['owner'], x['project'],
                    x['stack'], x['stack'])), axis=1)
    df['project'] = df.apply(
        lambda x: ("<a href='/rendersource/%s/owner/%s/project/%s'> %s </a>" %
                   (rendersource, x['owner'], x['project'], x['project'])),
        axis=1)
    df['owner'] = df.apply(
        lambda x: ("<a href='/rendersource/%s/owner/%s'> %s </a>" %
                   (rendersource, x['owner'], x['owner'])),
        axis=1)
    return df


@app.route('/rendersource/<path:rendersource>/owner/<owner>')
def owner_view(rendersource, owner):
    host, port = get_host_port(rendersource)
    render = renderapi.render.connect(host, port, owner, '', client_scripts="")
    projects = renderapi.render.get_projects_by_owner(render=render)
    stackInfo = renderapi.render.get_stack_metadata_by_owner(render=render)
    df = stackInfo_to_df(stackInfo, rendersource)

    return render_template('owner.html',
                           rendersource=rendersource,
                           owner=owner,
                           projects=projects,
                           tables=[df.to_html(escape=False)],
                           titles=['stacks'])


@app.route('/rendersource/<path:rendersource>')
def rendersource_index(rendersource):
    host, port = get_host_port(rendersource)
    owners = renderapi.render.get_owners(host=host, port=port)
    return render_template(
        'render.html', rendersource=rendersource, owners=owners)


@app.route('/rendersource/<path:rendersource>/owner/<owner>/project/<project>'
           '/stack/<stack>/collection/<collection>/pointmatch'
           '/z1/<int:z1>/z2/<int:z2>')
def pointmatch_z1_z2(rendersource, owner, project, stack, collection, z1, z2):
    try:
        host, port = get_host_port(rendersource)
        render = renderapi.render.connect(
            host, port, owner, project, client_scripts='')
        fig = make_pointmatch_plot(render, stack, collection, z1, z2)
        js_resources = INLINE.render_js()
        css_resources = INLINE.render_css()

        script, div = components(fig)
        html = render_template(
            'embed.html',
            plot_script=script,
            plot_div=div,
            js_resources=js_resources,
            css_resources=css_resources
        )

        return html

    except renderapi.errors.RenderError as r:
        return r.message
    return ("host:%s port:%d rendersource:%s owner:%s project:%s "
            "stack:%s collection:%s" % (
                host, port, rendersource, owner, project, stack, collection))


if __name__ == '__main__':
    app.run()
