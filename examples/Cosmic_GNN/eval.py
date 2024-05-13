import visualization_utils as vu

dir = '/mnt/data1/karres/cosmics_test'

print('Plotting efficiencies and purities')

eff_cosmic, pur_cosmic = vu.get_eff_pur_multi(dir, 'feature_store_cosmic', ['fc_uncut', 'fc_both_cut', 'fc_ladder_cut'])
vu.plot_eff_pur(eff_cosmic, pur_cosmic, ['FC uncut', 'FC w/ both cuts', 'FC w/ ladder cut'], 'Cosmics')
print('Cosmics done')

eff_cosmic_michel, pur_cosmic_michel = vu.get_eff_pur_multi(dir, 'feature_store_cosmic_michel', ['fc_uncut', 'fc_both_cut', 'fc_ladder_cut'])
vu.plot_eff_pur(eff_cosmic_michel, pur_cosmic_michel, ['FC uncut', 'FC w/ both cuts', 'FC w/ ladder cut'], 'Cosmics with Michel')
print('Cosmics with Michel done')

print('Plotting distances and angles')

dist_cosmic, ang_cosmic = vu.get_distances_and_angles_multi(dir, 'feature_store_cosmic', ['truth_graphs', 'fc_uncut', 'fc_both_cut', 'fc_ladder_cut'])
vu.plot_distance_angle(dist_cosmic, ang_cosmic, ['Truth', 'FC uncut', 'FC w/ both cuts', 'FC w/ ladder cut'], 'Cosmics')
print('Cosmics done')

dist_cosmic_michel, ang_cosmic_michel = vu.get_distances_and_angles_multi(dir, 'feature_store_cosmic_michel', ['truth_graphs', 'fc_uncut', 'fc_both_cut', 'fc_ladder_cut'])
vu.plot_distance_angle(dist_cosmic_michel, ang_cosmic_michel, ['Truth', 'FC uncut', 'FC w/ both cuts', 'FC w/ ladder cut'], 'Cosmics with Michel')
print('Cosmics with Michel done')