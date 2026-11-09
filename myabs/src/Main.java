import java.io.*;
import java.nio.file.*;
import java.util.*;

public class Main {

    public static void main(String[] args) throws Exception {
        if (args.length < 3 || !(args[2].equals("small") || args[2].equals("medium"))) {
            System.err.println("How to use: java Main <path_buggy> <path_fixed> <small|medium>");
            return;
        }

        Path buggyDir = Paths.get(args[0]), fixedDir = Paths.get(args[1]);
        if (!Files.isDirectory(buggyDir) || !Files.isDirectory(fixedDir)) {
            System.err.println("Not found buggy or fixed. Check out the path agian!");
            return;
        }

        Path outDir = Paths.get("out", args[2]);                
        Path idiomsPath = Paths.get("idioms", "idioms.csv");   
        if (!Files.exists(idiomsPath)) {
            System.err.println("Not found idioms/idioms.csv");
            return;
        }
        Files.createDirectories(outDir);

        Set<String> idioms = new HashSet<>(Files.readAllLines(idiomsPath));

        List<String> names = new ArrayList<>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(buggyDir, "*.java")) {
            for (Path p : ds) names.add(p.getFileName().toString());
        }

        names.sort(Comparator.comparingInt(Main::numKey));
        if (args.length > 3) names = names.subList(0, Math.min(Integer.parseInt(args[3]), names.size()));

        try (BufferedWriter wb = Files.newBufferedWriter(outDir.resolve("buggy_all.txt"));
             BufferedWriter wf = Files.newBufferedWriter(outDir.resolve("fixed_all.txt"))) {
            int n = 0;
            for (String name : names) {
                String b = PreProcessor.clean(new String(Files.readAllBytes(buggyDir.resolve(name))));
                String f = PreProcessor.clean(new String(Files.readAllBytes(fixedDir.resolve(name))));

                AstCollector ast = new AstCollector();
                try {
                    ast.parse(b);
                    ast.parse(f);
                } catch (StackOverflowError e) {
                    System.err.println("StackOverflow during parsing: " + name);
                } catch (Exception e) {
                    System.err.println("Parsing ERROR: " + name);
                }

                Abstractor abs = new Abstractor(idioms, ast.getTypes(), ast.getMethods());
                String ob = abs.tokenize(b);
                String of = abs.tokenize(f);

                wb.write(ob + " \n");
                wf.write(of + " \n");
                if (++n % 1000 == 0) System.out.println(n);
            }
        }
        System.out.println("Done!");
    }

    public static int numKey(String name) {
        String d = name.replaceAll("\\D", "");
        if (d.isEmpty()) return 0;
        return Integer.parseInt(d);
    }
}