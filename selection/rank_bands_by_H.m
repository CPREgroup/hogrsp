function [band_ranking, band_scores] = rank_bands_by_H(H)

band_scores = sqrt(sum(H.^2, 2));

[~, band_ranking] = sort(band_scores, 'descend');

end