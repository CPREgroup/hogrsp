function cube_norm = normalize_cube(cube)

cube = double(cube);

min_val = min(cube(:));
max_val = max(cube(:));

if max_val - min_val < eps
    cube_norm = cube;
else
    cube_norm = (cube - min_val) / (max_val - min_val);
end

end