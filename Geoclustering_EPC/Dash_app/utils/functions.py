import dash_leaflet as dl
import dash_leaflet.express as dlx

def filtering_dataset(df_, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13 ):
        df= df_.copy()
        df['Interfloor'] = df['heated_gross_volume'] / df['heated_usable_area']

        # FILTER 1 - Interfloor
        if f1:
            # Interfloor: the interfloor should be higher than 2.60 meter
            df = df.loc[df['Interfloor']>=2.80, :]

        if f2:
            #FILTER 2 - System efficiency
            rendimento = []
            for i, row in df.iterrows():
                # print(round(float(row['QHnd'])/(float(row['EPh'])*float(row['superficie_Utile_Riscaldata'])),3))
                try: 
                    rend_bui = round(float(row['QHimp'])/(float(row['EPh'])*float(row['heated_usable_area'])),3)
                except:
                    rend_bui = -1

                rendimento.append(rend_bui)
            df['system_performance'] = rendimento
            # value 0.566 is the minimum value given by the product of the efficiencies of different plant substitutions ( control, emission,distribution, generation)
            df = df.loc[(df['system_performance'] > 0.566), :]

        if f3:
            # FILTER 3 - EPh and QHnd
            # Removing EPC with EPh higher than 0 and QHnd higher than 0
            df = df.loc[(df['EPh']!=0) & (df['QHnd']!=0),:]

        if f4:
            # FILTER 4 - WIndow Surface
            #Removing buildings with window surface = 0
            df = df.loc[df['total_glazed_surface']!=0,:]

        if f5:
            # FILTER 5 - Opaque Surface
            # removing building with Opaque surface = 0
            df = df.loc[df['total_opaque_surface']!=0,:]

        if f6:
            # FILTER 6 - Building Geometry 
            # Geometric condition: we consider a building plan as a corridor with width of 1.5 meter (as a corridor) and roof with slope of 30%
            # The surface of the parallelepiped with width 1.5 and interfloor defined in the point 1 should be hifher than the sum of Opaque and Transparet Surface
            df['corridor'] = (df['heated_usable_area']/1.5 + 1.5)*2*df['Interfloor']+df['heated_usable_area']+(df['heated_usable_area']/0.86)
            df = df.loc[df['corridor']>(df['total_glazed_surface']+df['total_opaque_surface'])]

        if f7:
            # FILTER 7 - aeroilluminant ratio
            # aeroilluminant ratio: the transparent surface of the facade should be higher than the aeroilluminant ratio:
            rapporto_aereo = []
            for surface in  df['heated_usable_area']:
                rapporto_aereo.append(surface/30)
            df['aereo'] = rapporto_aereo 
            df = df.loc[df['total_glazed_surface']>df['aereo'], :]

        if f8:
            # FILTER 8 - Window Transmittance
            # The Transmittance of the window should be between 0.6 and 8
            df = df.loc[(df['average_glazed_surface_transmittance'] >= 0.6) & (df['average_glazed_surface_transmittance'] <= 8), :]

        if f9:
            # FILTER 9 - Theoretical power generator
            #Theoretical power of the plant must be lower than the rated power. The theoretical power is calculated as the propduct of the opaque surface with its # average transmittance + the transparent surface with its average transmittance all multiplied by delta T = 20°C in kW
            df['theoric_nominal_power'] = (df['total_opaque_surface']*df['average_opaque_surface_transmittance']+df['total_glazed_surface']*df['average_glazed_surface_transmittance'])*20/1000
            df = df.loc[df['nominal_power'] >= df['theoric_nominal_power'],:]

        if f10:
            # FILTER 10 - Solar Area
            # Solar Area should be lower than the transparent surface
            df = df.loc[df['Asol'] <= df['total_opaque_surface'],:]

        if f11:
            # FILTER 11 - year of building and system
            # The year of. construction of the building must be less than or equal to the year of construction of the facility + a safety factor = 3
            df = df.loc[df['construction_year']<= df['installation_year']+3,:]

        if f12:
            # FILTER 12: Minimum building surface
            # Based on the intended use of the building a minimum heating/cooling surface is considered: 
            # '''
            # 1 = E.1 (1) dwellings used for residence with continuous occupation, such as civil and rural dwellings, boarding schools, convents, penalty houses, barracks;
            # 2 = E.1 (2) dwellings used as residences with occasional occupation, such as vacation homes, weekend homes and the like;
            # 3 = E.1 (3) buildings used for hotel, boarding house and similar activities;

            # 4 = E.2 Office and similar buildings: public or private, independent or contiguous to buildings also used for industrial or craft activities, provided that they are separable from such buildings for the purposes of thermal insulation;

            # 5 = E.3 Buildings used as hospitals, clinics or nursing homes and assimilated including those used for the hospitalization or care of minors or the elderly as well as sheltered facilities for the care and recovery of drug addicts and other persons entrusted to public social services;

            # 6 = E.4 Buildings used for recreation or worship and similar activities:

            # 7 = E.4 (1) such as cinemas and theaters, conference meeting rooms;
            # 8 = E.4 (2) such as exhibitions, museums and libraries, places of worship;
            # 9 = E.4 (3) such as bars, restaurants, dance halls;

            # 10 = E.5 Buildings used for commercial and similar activities: such as stores, wholesale or retail warehouses, supermarkets, exhibitions;

            # 11 = E.6 Buildings used for sports activities:

            # 12 = E.6 (1) swimming pools, saunas and similar;
            # 13 = E.6 (2) gymnasiums and similar;
            # 14 = E.6 (3) support services for sports activities;

            # Uffici 
            thresholds = {
                1: 80,
                2: 40,
                3: 100,
                4: 40,
                5: 200,
                6: 80,
                7: 80,
                8: 80,
                9: 50,
                10: 30,
                11: 100,
                12: 100,
                13: 100,
                14: 50,
                15: 80,  # Consider lowering this value for schools if needed
                16: 100
            }

            check_area = []
            for i,classificazione in enumerate(df['DPR412_classification']):
                print()
                # print(df.at[i, 'superficie_Netta'])
                # Get the threshold for the current classification
                threshold = thresholds.get(classificazione, None)

                if threshold is not None:
                    # Check the area against the threshold
                    if df.iloc[i, :]['heated_usable_area'] < threshold:
                        check_area.append(False)
                    else:
                        check_area.append(True)
                else:
                    # Handle cases where the classification is not in the thresholds dictionary
                    raise ValueError(f"Unknown classification: {classificazione}")

            df['Area_check'] = check_area
            df = df.loc[df['Area_check'] == True,]

        if f13:
            # FILTER 13: Air change rate
            # Air changes must be at least greater than 0.3, and estimating a maximum air change taking into account a crowding index of 1.5 (Table VIII, Appendix # A of 10339 standard) equal to the highest value for spectator areas in buildings used for sports activities and 39m3/h equal to 11liters/second per # person, the changes calculated by the EPC must be less than these values

            df = df.loc[df['air_changes']>0.3,:]
            df['air_changes_max'] = ((df['heated_usable_area']*1.5)*39.6)/df['heated_gross_volume']
            df = df.loc[df['air_changes_max'] >= df['air_changes'],:]

        return df

color_map = {
    0:"black", 1: "red", 2: "blue", 3: "green", 4: "purple", 5: "orange",
    6: "pink", 7: "brown", 8: "gray", 9: "black", 10: "cyan",
    11: "lime", 12: "navy", 13: "gold"
}

def create_map(data_map, height_map:str='50vh', id_map:str="map_"):
    markers = [
        dl.CircleMarker(
            center=[row["lat"], row["lon"]],
            radius=4,  # Marker size
            color="transparent",  # Border color
            fillColor=color_map[int(row["variable"])],  # Fill color based on category
            fillOpacity=0.8,  # Transparency
            children=[dl.Tooltip(f"Category: {row['variable']}")]
        ) for _, row in data_map.iterrows()
    ]
    initial_center = [45.02569105418987, 7.671092180850915]
    Map = dl.Map(
        center=initial_center,
        zoom=6,
        maxZoom=12,
        children = [
            dl.TileLayer(),
            dl.LayerGroup(markers),
        ],
        id=id_map,
        style={'width': '100%', 'height': height_map, 'zIndex':0}
    )

    return Map