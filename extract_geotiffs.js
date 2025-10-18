var dataset = ee.ImageCollection('NASA/VIIRS/002/VNP46A2')
                  .filter(ee.Filter.date('2020-01-01', '2020-12-31'));

var nighttime = dataset.select('Gap_Filled_DNB_BRDF_Corrected_NTL');

// Apply logarithmic transformation
// Add 1 to avoid log(0) issues
var nighttimeLog = nighttime.map(function(image) {
  return ee.Image(1).add(image).log10();
});

// Create median composite
var nighttimeLogMedian = nighttimeLog.median();

// Wavelength color palette (visible spectrum)
var wavelengthPalette = [
  '000000', '000000', '000000', '000000', '000000',
  '000000', '000000', '000000', '000000', '000000',
  '000000', '000000', '000000', '000000', '000000',
  '000000', '000000', '000000', '000000', '000000',
  '000001', '000001', '000001', '000001', '010001',
  '010001', '010001', '010001', '010001', '010001',
  '010001', '010002', '010002', '010002', '010002',
  '010002', '020002', '020002', '020003', '020003',
  '020003', '020003', '020003', '020004', '030004',
  '030004', '030004', '030005', '030005', '030005',
  '030005', '040006', '040006', '040006', '040006',
  '040007', '050007', '050007', '050008', '050008',
  '050008', '060009', '060009', '060009', '06000a',
  '07000a', '07000b', '07000b', '08000b', '08000c',
  '08000c', '08000d', '09000d', '09000e', '09000e',
  '0a000e', '0a000f', '0a000f', '0b0010', '0b0010',
  '0b0011', '0c0012', '0c0012', '0c0013', '0d0013',
  '0d0014', '0e0014', '0e0015', '0e0016', '0f0016',
  '0f0017', '100017', '100018', '100019', '110019',
  '11001a', '12001b', '12001c', '13001c', '13001d',
  '14001e', '14001e', '15001f', '150020', '160021',
  '160022', '170022', '170023', '180024', '180025',
  '190026', '1a0027', '1a0027', '1b0028', '1b0029',
  '1c002a', '1d002b', '1d002c', '1e002d', '1e002e',
  '1f002f', '200030', '200031', '210032', '220033',
  '220034', '230035', '240036', '240037', '250038',
  '260039', '27003a', '27003b', '28003d', '29003e',
  '2a003f', '2a0040', '2b0041', '2c0042', '2d0044',
  '2e0045', '2e0046', '2f0047', '300049', '31004a',
  '32004b', '33004d', '34004e', '34004f', '350051',
  '360052', '370053', '380055', '390056', '3a0057',
  '3b0059', '3c005a', '3d005c', '3e005d', '3f005f',
  '400060', '410062', '420063', '430065', '440066',
  '450068', '460069', '47006b', '48006d', '49006e',
  '4a0070', '4b0072', '4c0073', '4e0075', '4f0077',
  '500078', '51007a', '52007c', '53007e', '54007f',
  '560081', '570083', '580085', '590087', '5a0088',
  '5c008a', '5d008c', '5e008e', '5f0090', '610092',
  '620094', '630096', '650098', '66009a', '67009c',
  '69009e', '6a00a0', '6b00a2', '6d00a4', '6e00a6',
  '6e00a6', '6f00a6', '6f00a7', '6f00a7', '6f00a8',
  '6f00a8', '6f00a9', '6f00a9', '6f00aa', '6f00aa',
  '6f00ab', '6f00ab', '6f00ac', '6f00ac', '6f00ad',
  '6f00ad', '6f00ae', '6f00ae', '6f00af', '6f00af',
  '6f00af', '6f00b0', '6f00b0', '6f00b1', '6f00b1',
  '6f00b2', '6f00b2', '6e00b3', '6e00b3', '6e00b4',
  '6e00b4', '6e00b5', '6e00b5', '6e00b6', '6e00b6',
  '6e00b7', '6e00b7', '6e00b8', '6e00b8', '6e00b9',
  '6e00b9', '6e00ba', '6e00ba', '6e00bb', '6e00bb',
  '6e00bc', '6e00bc', '6e00bd', '6e00bd', '6e00be',
  '6d00be', '6d00bf', '6d00bf', '6d00c0', '6d00c0',
  '6d00c1', '6d00c1', '6d00c2', '6d00c2', '6d00c3',
  '6d00c3', '6d00c3', '6d00c4', '6c00c4', '6c00c5',
  '6c00c5', '6c00c6', '6c00c6', '6c00c7', '6c00c7',
  '6c00c8', '6c00c8', '6c00c9', '6b00c9', '6b00ca',
  '6b00ca', '6b00cb', '6b00cb', '6b00cc', '6b00cc',
  '6b00cd', '6b00cd', '6a00ce', '6a00ce', '6a00cf',
  '6a00cf', '6a00d0', '6a00d0', '6a00d1', '6a00d1',
  '6900d2', '6900d2', '6900d3', '6900d3', '6900d4',
  '6900d4', '6900d5', '6800d5', '6800d6', '6800d6',
  '6800d6', '6800d7', '6800d7', '6700d8', '6700d8',
  '6700d9', '6700d9', '6700da', '6700da', '6600db',
  '6600db', '6600dc', '6600dc', '6600dd', '6600dd',
  '6500de', '6500de', '6500df', '6500df', '6500e0',
  '6500e0', '6400e1', '6400e1', '6400e2', '6400e2',
  '6400e3', '6300e3', '6300e4', '6300e4', '6300e5',
  '6300e5', '6200e6', '6200e6', '6200e7', '6200e7',
  '6100e8', '6100e8', '6100e9', '6100e9', '6100e9',
  '6000ea', '6000ea', '6000eb', '6000eb', '5f00ec',
  '5f00ec', '5f00ed', '5f00ed', '5f00ee', '5e00ee',
  '5e00ef', '5e00ef', '5e00f0', '5d00f0', '5d00f1',
  '5d00f1', '5d00f2', '5c00f2', '5c00f3', '5c00f3',
  '5b00f4', '5b00f4', '5b00f5', '5b00f5', '5a00f6',
  '5a00f6', '5a00f7', '5a00f7', '5900f8', '5900f8',
  '5900f9', '5900f9', '5800fa', '5800fa', '5800fb',
  '5700fb', '5700fc', '5700fc', '5700fd', '5600fd',
  '5600fd', '5600fe', '5500fe', '5500ff', '5500ff',
  '5400ff', '5400ff', '5300ff', '5300ff', '5200ff',
  '5200ff', '5100ff', '5100ff', '5000ff', '5000ff',
  '4f00ff', '4f00ff', '4f00ff', '4e00ff', '4e00ff',
  '4d00ff', '4d00ff', '4c00ff', '4c00ff', '4b00ff',
  '4b00ff', '4a00ff', '4a00ff', '4900ff', '4900ff',
  '4900ff', '4800ff', '4800ff', '4700ff', '4700ff',
  '4600ff', '4600ff', '4500ff', '4500ff', '4400ff',
  '4400ff', '4300ff', '4300ff', '4200ff', '4200ff',
  '4200ff', '4100ff', '4100ff', '4000ff', '4000ff',
  '3f00ff', '3f00ff', '3e00ff', '3e00ff', '3d00ff',
  '3d00ff', '3c00ff', '3c00ff', '3c00ff', '3b00ff',
  '3b00ff', '3a00ff', '3a00ff', '3900ff', '3900ff',
  '3800ff', '3800ff', '3700ff', '3700ff', '3600ff',
  '3600ff', '3500ff', '3500ff', '3500ff', '3400ff',
  '3400ff', '3300ff', '3300ff', '3200ff', '3200ff',
  '3100ff', '3100ff', '3000ff', '3000ff', '2f00ff',
  '2f00ff', '2f00ff', '2e00ff', '2e00ff', '2d00ff',
  '2d00ff', '2c00ff', '2c00ff', '2b00ff', '2b00ff',
  '2a00ff', '2a00ff', '2900ff', '2900ff', '2800ff',
  '2800ff', '2800ff', '2700ff', '2700ff', '2600ff',
  '2600ff', '2500ff', '2500ff', '2400ff', '2400ff',
  '2300ff', '2300ff', '2200ff', '2200ff', '2200ff',
  '2100ff', '2100ff', '2000ff', '2000ff', '1f00ff',
  '1f00ff', '1e00ff', '1e00ff', '1d00ff', '1d00ff',
  '1c00ff', '1c00ff', '1b00ff', '1b00ff', '1b00ff',
  '1a00ff', '1a00ff', '1900ff', '1900ff', '1800ff',
  '1800ff', '1700ff', '1700ff', '1600ff', '1600ff',
  '1500ff', '1500ff', '1500ff', '1400ff', '1400ff',
  '1300ff', '1300ff', '1200ff', '1200ff', '1100ff',
  '1100ff', '1000ff', '1000ff', '0f00ff', '0f00ff',
  '0e00ff', '0e00ff', '0e00ff', '0d00ff', '0d00ff',
  '0c00ff', '0c00ff', '0b00ff', '0b00ff', '0a00ff',
  '0a00ff', '0900ff', '0900ff', '0800ff', '0800ff',
  '0800ff', '0700ff', '0700ff', '0600ff', '0600ff',
  '0500ff', '0500ff', '0400ff', '0400ff', '0300ff',
  '0300ff', '0200ff', '0200ff', '0100ff', '0100ff',
  '0100ff', '0000ff', '0000ff', '0001ff', '0002ff',
  '0002ff', '0003ff', '0003ff', '0004ff', '0004ff',
  '0005ff', '0005ff', '0006ff', '0007ff', '0007ff',
  '0008ff', '0008ff', '0009ff', '0009ff', '000aff',
  '000aff', '000bff', '000cff', '000cff', '000dff',
  '000dff', '000eff', '000eff', '000fff', '000fff',
  '0010ff', '0011ff', '0011ff', '0012ff', '0012ff',
  '0013ff', '0013ff', '0014ff', '0014ff', '0015ff',
  '0016ff', '0016ff', '0017ff', '0017ff', '0018ff',
  '0018ff', '0019ff', '001aff', '001aff', '001bff',
  '001bff', '001cff', '001cff', '001dff', '001dff',
  '001eff', '001fff', '001fff', '0020ff', '0020ff',
  '0021ff', '0021ff', '0022ff', '0022ff', '0023ff',
  '0024ff', '0024ff', '0025ff', '0025ff', '0026ff',
  '0026ff', '0027ff', '0027ff', '0028ff', '0029ff',
  '0029ff', '002aff', '002aff', '002bff', '002bff',
  '002cff', '002cff', '002dff', '002eff', '002eff',
  '002fff', '002fff', '0030ff', '0030ff', '0031ff',
  '0031ff', '0032ff', '0033ff', '0033ff', '0034ff',
  '0034ff', '0035ff', '0035ff', '0036ff', '0036ff',
  '0037ff', '0038ff', '0038ff', '0039ff', '0039ff',
  '003aff', '003aff', '003bff', '003bff', '003cff',
  '003dff', '003dff', '003eff', '003eff', '003fff',
  '003fff', '0040ff', '0040ff', '0041ff', '0042ff',
  '0042ff', '0043ff', '0043ff', '0044ff', '0044ff',
  '0045ff', '0046ff', '0046ff', '0047ff', '0047ff',
  '0048ff', '0048ff', '0049ff', '0049ff', '004aff',
  '004bff', '004bff', '004cff', '004cff', '004dff',
  '004dff', '004eff', '004eff', '004fff', '0050ff',
  '0050ff', '0051ff', '0051ff', '0052ff', '0052ff',
  '0053ff', '0053ff', '0054ff', '0055ff', '0055ff',
  '0056ff', '0056ff', '0057ff', '0057ff', '0058ff',
  '0058ff', '0059ff', '005aff', '005aff', '005bff',
  '005bff', '005cff', '005cff', '005dff', '005dff',
  '005eff', '005fff', '005fff', '0060ff', '0060ff',
  '0061ff', '0061ff', '0062ff', '0062ff', '0063ff',
  '0064ff', '0064ff', '0065ff', '0065ff', '0066ff',
  '0066ff', '0067ff', '0067ff', '0068ff', '0069ff',
  '0069ff', '006aff', '006aff', '006bff', '006bff',
  '006cff', '006cff', '006dff', '006eff', '006eff',
  '006fff', '006fff', '0070ff', '0070ff', '0071ff',
  '0072ff', '0072ff', '0073ff', '0073ff', '0074ff',
  '0074ff', '0075ff', '0075ff', '0076ff', '0077ff',
  '0077ff', '0078ff', '0078ff', '0079ff', '0079ff',
  '007aff', '007aff', '007bff', '007cff', '007cff',
  '007dff', '007dff', '007eff', '007eff', '007fff',
  '007fff', '0080ff', '0081ff', '0081ff', '0082ff',
  '0082ff', '0083ff', '0083ff', '0084ff', '0084ff',
  '0085ff', '0086ff', '0086ff', '0087ff', '0087ff',
  '0088ff', '0088ff', '0089ff', '0089ff', '008aff',
  '008bff', '008bff', '008cff', '008cff', '008dff',
  '008dff', '008eff', '008eff', '008fff', '0090ff',
  '0090ff', '0091ff', '0091ff', '0092ff', '0092ff',
  '0093ff', '0093ff', '0094ff', '0095ff', '0095ff',
  '0096ff', '0096ff', '0097ff', '0097ff', '0098ff',
  '0099ff', '0099ff', '009aff', '009aff', '009bff',
  '009bff', '009cff', '009cff', '009dff', '009eff',
  '009eff', '009fff', '009fff', '00a0ff', '00a0ff',
  '00a1ff', '00a1ff', '00a2ff', '00a3ff', '00a3ff',
  '00a4ff', '00a4ff', '00a5ff', '00a5ff', '00a6ff',
  '00a6ff', '00a7ff', '00a8ff', '00a8ff', '00a9ff',
  '00a9ff', '00aaff', '00aaff', '00abff', '00abff',
  '00acff', '00adff', '00adff', '00aeff', '00aeff',
  '00afff', '00afff', '00b0ff', '00b0ff', '00b1ff',
  '00b2ff', '00b2ff', '00b3ff', '00b3ff', '00b4ff',
  '00b4ff', '00b5ff', '00b5ff', '00b6ff', '00b7ff',
  '00b7ff', '00b8ff', '00b8ff', '00b9ff', '00b9ff',
  '00baff', '00baff', '00bbff', '00bcff', '00bcff',
  '00bdff', '00bdff', '00beff', '00beff', '00bfff',
  '00bfff', '00c0ff', '00c1ff', '00c1ff', '00c2ff',
  '00c2ff', '00c3ff', '00c3ff', '00c4ff', '00c5ff',
  '00c5ff', '00c6ff', '00c6ff', '00c7ff', '00c7ff',
  '00c8ff', '00c8ff', '00c9ff', '00caff', '00caff',
  '00cbff', '00cbff', '00ccff', '00ccff', '00cdff',
  '00cdff', '00ceff', '00cfff', '00cfff', '00d0ff',
  '00d0ff', '00d1ff', '00d1ff', '00d2ff', '00d2ff',
  '00d3ff', '00d4ff', '00d4ff', '00d5ff', '00d5ff',
  '00d6ff', '00d6ff', '00d7ff', '00d7ff', '00d8ff',
  '00d9ff', '00d9ff', '00daff', '00daff', '00dbff',
  '00dbff', '00dcff', '00dcff', '00ddff', '00deff',
  '00deff', '00dfff', '00dfff', '00e0ff', '00e0ff',
  '00e1ff', '00e1ff', '00e2ff', '00e3ff', '00e3ff',
  '00e4ff', '00e4ff', '00e5ff', '00e5ff', '00e6ff',
  '00e6ff', '00e7ff', '00e8ff', '00e8ff', '00e9ff',
  '00e9ff', '00eaff', '00eaff', '00ebff', '00ecff',
  '00ecff', '00edff', '00edff', '00eeff', '00eeff',
  '00efff', '00efff', '00f0ff', '00f1ff', '00f1ff',
  '00f2ff', '00f2ff', '00f3ff', '00f3ff', '00f4ff',
  '00f4ff', '00f5ff', '00f6ff', '00f6ff', '00f7ff',
  '00f7ff', '00f8ff', '00f8ff', '00f9ff', '00f9ff',
  '00faff', '00fbff', '00fbff', '00fcff', '00fcff',
  '00fdff', '00fdff', '00feff', '00feff', '00ffff'
];

// Visualization parameters (same for all exports)
var nighttimeLogVis = {
  min: 0,           // log10(1) for original value 0
  max: 4.27,        // log10(18581) for original value 18580
  palette: wavelengthPalette
};

// Apply visualization to create RGB image for export
var nighttimeRGB = nighttimeLogMedian.visualize({
  min: nighttimeLogVis.min,
  max: nighttimeLogVis.max,
  palette: wavelengthPalette
});

// Democratic Republic of Congo (DRC) - reference size at equator
// DRC is approximately 19.11° × 18.85°
var drc = ee.Geometry.Rectangle([
  12.20,    // west
  -13.46,   // south
  31.31,    // east (19.11° width)
  5.39      // north (18.85° height)
]);

// US - Moline, IL at west (-90.5°), Montreal at north (45.5°)
// Center ~36°N, cos(36°) ≈ 0.809, need width/0.809 ≈ 23.6°
var us = ee.Geometry.Rectangle([
  -90.5,    // west (Moline, Illinois)
  26.65,    // south (45.5 - 18.85)
  -66.9,    // east (23.6° width for latitude compensation)
  45.5      // north (Montreal)
]);

// China - Adjusted with proper latitude compensation
// Center ~30°N, cos(30°) ≈ 0.866, need width/0.866
var china = ee.Geometry.Rectangle([
  109.5,    // west
  21.0,     // south
  131.5,    // east (22° width to compensate for ~30°N latitude)
  39.85     // north (18.85° height)
]);

// Europe - Adjusted with proper latitude compensation
// Center ~51°N, cos(51°) ≈ 0.629, need width/0.629
// Moved center 250km (~3.6°) west
var europe = ee.Geometry.Rectangle([
  -11.6,    // west (moved 3.6° west from -8.0)
  42.0,     // south
  18.8,     // east (moved 3.6° west from 22.4)
  60.85     // north (18.85° height)
]);

// Netherlands frames
// 1. Full Netherlands
// Extended 800km westward to west border of Ireland
var netherlands_full = ee.Geometry.Rectangle([
  -10.5,    // west (extended to west border of Ireland)
  50.75,    // south (Limburg)
  7.2,      // east (German border - unchanged)
  53.5      // north (Wadden Islands)
]);

// 2. Regional: Rotterdam harbour, Westland, Moerkappele, Den Bosch
// Westland: ~52.0°N, 4.2°E
// Rotterdam Maasvlakte: ~51.95°N, 4.0°E
// Moerkappele: ~52.046°N, 4.565°E
// Den Bosch: ~51.7°N, 5.3°E
var netherlands_regional = ee.Geometry.Rectangle([
  3.8,      // west (west of Maasvlakte)
  51.5,     // south (south of Den Bosch)
  5.6,      // east (east of Den Bosch)
  52.3      // north (north of Moerkappele)
]);

// 3. Westland + Moerkappele focus
// Westland: ~52.0°N, 4.2°E
// Moerkappele: ~52.046°N, 4.565°E
var netherlands_westland_moerkappele = ee.Geometry.Rectangle([
  4.0,      // west
  51.8,     // south
  4.8,      // east
  52.25     // north
]);

// 4. Just Moerkappele (small area)
var netherlands_moerkappele = ee.Geometry.Rectangle([
  4.465,    // west
  51.996,   // south
  4.665,    // east (0.2° width)
  52.096    // north (0.1° height)
]);

Map.setCenter(-78.7, 36.0, 5);  // Center on US region
Map.addLayer(nighttimeLogMedian, nighttimeLogVis, 'Nighttime (Log Scale)');

// Calculate and print areas for validation
var drc_area = drc.area();
var us_area = us.area();
var china_area = china.area();
var europe_area = europe.area();

print('Area Validation (square meters):');
print('DRC area (target):', drc_area);
print('US area:', us_area);
print('China area:', china_area);
print('Europe area:', europe_area);
print('Target: ~4.46 trillion square meters');
print('US boundaries:');
print('- West: Moline, Illinois (-90.5°)');
print('- North: Montreal (45.5°)');
print('- Width: 23.6° (compensated for 36°N average latitude)');
print('- Height: 18.85°');

// Export DRC
Export.image.toDrive({
  image: nighttimeRGB,
  description: 'nighttime_drc_2020',
  folder: 'EarthEngine',
  fileNamePrefix: 'nighttime_drc_2020',
  region: drc,
  scale: 500,
  crs: 'EPSG:4326',
  fileFormat: 'GeoTIFF',
  maxPixels: 1e9
});

// Export US
Export.image.toDrive({
  image: nighttimeRGB,
  description: 'nighttime_us_2020',
  folder: 'EarthEngine',
  fileNamePrefix: 'nighttime_us_2020',
  region: us,
  scale: 500,
  crs: 'EPSG:4326',
  fileFormat: 'GeoTIFF',
  maxPixels: 1e9
});

// Export China
Export.image.toDrive({
  image: nighttimeRGB,
  description: 'nighttime_china_2020',
  folder: 'EarthEngine',
  fileNamePrefix: 'nighttime_china_2020',
  region: china,
  scale: 500,
  crs: 'EPSG:4326',
  fileFormat: 'GeoTIFF',
  maxPixels: 1e9
});

// Export Europe
Export.image.toDrive({
  image: nighttimeRGB,
  description: 'nighttime_europe_2020',
  folder: 'EarthEngine',
  fileNamePrefix: 'nighttime_europe_2020',
  region: europe,
  scale: 500,
  crs: 'EPSG:4326',
  fileFormat: 'GeoTIFF',
  maxPixels: 1e9
});

// Export Netherlands - Full
Export.image.toDrive({
  image: nighttimeRGB,
  description: 'nighttime_netherlands_full_2020',
  folder: 'EarthEngine',
  fileNamePrefix: 'nighttime_netherlands_full_2020',
  region: netherlands_full,
  scale: 500,
  crs: 'EPSG:4326',
  fileFormat: 'GeoTIFF',
  maxPixels: 1e9
});

// Export Netherlands - Regional
Export.image.toDrive({
  image: nighttimeRGB,
  description: 'nighttime_netherlands_regional_2020',
  folder: 'EarthEngine',
  fileNamePrefix: 'nighttime_netherlands_regional_2020',
  region: netherlands_regional,
  scale: 500,
  crs: 'EPSG:4326',
  fileFormat: 'GeoTIFF',
  maxPixels: 1e9
});

// Export Netherlands - Westland+Moerkappele
Export.image.toDrive({
  image: nighttimeRGB,
  description: 'nighttime_netherlands_westland_moerkappele_2020',
  folder: 'EarthEngine',
  fileNamePrefix: 'nighttime_netherlands_westland_moerkappele_2020',
  region: netherlands_westland_moerkappele,
  scale: 500,
  crs: 'EPSG:4326',
  fileFormat: 'GeoTIFF',
  maxPixels: 1e9
});

// Export Netherlands - Moerkappele only
Export.image.toDrive({
  image: nighttimeRGB,
  description: 'nighttime_netherlands_moerkappele_2020',
  folder: 'EarthEngine',
  fileNamePrefix: 'nighttime_netherlands_moerkappele_2020',
  region: netherlands_moerkappele,
  scale: 500,
  crs: 'EPSG:4326',
  fileFormat: 'GeoTIFF',
  maxPixels: 1e9
});

// Add all regions to map (all visible)
Map.addLayer(drc, {color: 'red'}, 'DRC', true);
Map.addLayer(us, {color: 'blue'}, 'US', true);
Map.addLayer(china, {color: 'green'}, 'China', true);
Map.addLayer(europe, {color: 'yellow'}, 'Europe', true);
Map.addLayer(netherlands_full, {color: 'orange'}, 'Netherlands Full', true);
Map.addLayer(netherlands_regional, {color: 'purple'}, 'Netherlands Regional', true);
Map.addLayer(netherlands_westland_moerkappele, {color: 'cyan'}, 'Netherlands Westland+Moerkappele', true);
Map.addLayer(netherlands_moerkappele, {color: 'magenta'}, 'Netherlands Moerkappele', true);

print('Export tasks created. Check the Tasks tab to run exports.');
print('Final configuration:');
print('- DRC: 19.11° × 18.85° (equatorial reference)');
print('- US: 23.6° × 18.85° (Moline to Atlantic, Florida to Montreal)');
print('- China: 22° × 18.85° (compensated for 30°N)');
print('- Europe: 30.4° × 18.85° (compensated for 51°N, center moved 250km west)');
print('- Netherlands Full: extended 800km westward to Ireland, includes actual Netherlands');
