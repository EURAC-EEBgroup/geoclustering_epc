-- \connect monitoring

-- --
-- -- Name: uuid-ossp; Type: EXTENSION; Schema: -; Owner: -
-- --

-- CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;


-- --
-- -- Name: EXTENSION "uuid-ossp"; Type: COMMENT; Schema: -; Owner: 
-- --

-- COMMENT ON EXTENSION "uuid-ossp" IS 'generate universally unique identifiers (UUIDs)';


-- SET TIMEZONE TO 'Europe/Rome';


-- -- add other fields?
-- CREATE TABLE public.building (
--     uuid uuid DEFAULT uuid_generate_v4() NOT NULL,
--     building_name character varying(50) NOT NULL,
--     PRIMARY KEY (uuid)
-- );




-- CREATE SEQUENCE public.sensor_id_seq
--     AS integer
--     START WITH 1
--     INCREMENT BY 1
--     NO MINVALUE
--     NO MAXVALUE
--     CACHE 1;

-- -- ALTER SEQUENCE sensor_id_seq OWNED BY sensor.id;

-- ALTER TABLE ONLY public.sensor ALTER COLUMN id SET DEFAULT nextval('public.sensor_id_seq'::regclass);

-- ALTER TABLE ONLY public.sensor
--     ADD CONSTRAINT fk_building_id FOREIGN KEY (building_id) REFERENCES public.building(uuid);



-- CREATE TABLE public.temperature (
--     time   TIMESTAMPTZ   NOT NULL,
--     building_id uuid not null,
--     sensor_id integer not null,
--     value       DOUBLE PRECISION  NOT NULL,
--     PRIMARY KEY (time, building_id, sensor_id)
-- );

-- CREATE TABLE public.humidity (
--     time   TIMESTAMPTZ   NOT NULL,
--     building_id uuid not null,
--     sensor_id integer not null,
--     value       DOUBLE PRECISION  NOT NULL,
--     PRIMARY KEY (time, building_id, sensor_id)
-- );

-- CREATE TABLE public.co2 (
--     time   TIMESTAMPTZ   NOT NULL,
--     building_id uuid not null,
--     sensor_id integer not null,
--     value       DOUBLE PRECISION  NOT NULL,
--     PRIMARY KEY (time, building_id, sensor_id)
-- );

-- ALTER TABLE ONLY public.temperature
--     ADD CONSTRAINT fk_building_id FOREIGN KEY (building_id) REFERENCES public.building(uuid);

-- ALTER TABLE ONLY public.humidity
--     ADD CONSTRAINT fk_building_id FOREIGN KEY (building_id) REFERENCES public.building(uuid);

-- ALTER TABLE ONLY public.co2
--     ADD CONSTRAINT fk_building_id FOREIGN KEY (building_id) REFERENCES public.building(uuid);

-- ALTER TABLE ONLY public.temperature
--     ADD CONSTRAINT fk_sensor_id FOREIGN KEY (sensor_id) REFERENCES public.sensor(id);

-- ALTER TABLE ONLY public.humidity
--     ADD CONSTRAINT fk_sensor_id FOREIGN KEY (sensor_id) REFERENCES public.sensor(id);

-- ALTER TABLE ONLY public.co2
--     ADD CONSTRAINT fk_sensor_id FOREIGN KEY (sensor_id) REFERENCES public.sensor(id);


-- SELECT create_hypertable('public.temperature', 'time', if_not_exists => TRUE, create_default_indexes => TRUE);

-- SELECT create_hypertable('public.humidity', 'time', if_not_exists => TRUE, create_default_indexes => TRUE);

-- SELECT create_hypertable('public.co2', 'time', if_not_exists => TRUE, create_default_indexes => TRUE);

